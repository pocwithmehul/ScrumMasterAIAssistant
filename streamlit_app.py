"""
Scrum Master AI Assistant — Streamlit local dashboard.

Runs the observability pipeline directly in-process (no HTTP server needed).
Launch via:
    streamlit run streamlit_app.py
  or:
    python main.py --localgui
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import streamlit as st
import yaml

# ── make the installed package importable when run from project root ──────────
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scrum_master_assistant.llmops.tracker import get_tracker
from scrum_master_assistant.models.config import AppSettings
from scrum_master_assistant.runtime.factory import build_pipeline

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scrum Master Assistant",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
_FLAG_OVERRIDES_FILE = Path("config/ui-flag-overrides.json")
_UI_SETTINGS_FILE = Path("config/ui-settings.yml")

_FLAG_GROUPS = {
    "Data Source Connectors": [
        "source.datadog", "source.aws_cloudwatch", "source.aws_cloudtrail",
        "source.azure_monitor", "source.gcp_observability", "source.splunk",
        "source.elasticsearch", "source.aws_opensearch",
    ],
    "Dashboard Views": [
        "dashboard.security", "dashboard.api",
        "dashboard.infrastructure", "dashboard.certificates",
    ],
}

_SOURCE_SETTINGS: dict[str, dict[str, str]] = {
    "Datadog": {
        "datadog_site": "Site",
        "datadog_log_query": "Log Query",
        "datadog_hours_lookback": "Lookback Hours",
        "datadog_page_limit": "Page Limit",
    },
    "AWS CloudWatch": {
        "cloudwatch_hours_lookback": "Lookback Hours",
    },
    "AWS CloudTrail": {
        "cloudtrail_hours_lookback": "Lookback Hours",
    },
    "Azure Monitor": {
        "azure_monitor_hours_lookback": "Lookback Hours",
    },
    "GCP Observability": {
        "gcp_logging_hours_lookback": "Lookback Hours",
    },
    "Splunk": {
        "splunk_base_url": "Base URL",
        "splunk_search_query": "Search Query",
        "splunk_hours_lookback": "Lookback Hours",
        "splunk_max_results": "Max Results",
    },
    "Elasticsearch": {
        "elasticsearch_url": "URL",
        "elasticsearch_index": "Index Pattern",
        "elasticsearch_query": "Query",
        "elasticsearch_hours_lookback": "Lookback Hours",
        "elasticsearch_max_results": "Max Results",
    },
    "AWS OpenSearch": {
        "opensearch_url": "Endpoint URL",
        "opensearch_region": "AWS Region",
        "opensearch_index": "Index Pattern",
        "opensearch_hours_lookback": "Lookback Hours",
        "opensearch_max_results": "Max Results",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_flag_overrides() -> dict[str, bool]:
    if _FLAG_OVERRIDES_FILE.exists():
        try:
            return json.loads(_FLAG_OVERRIDES_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_flag_overrides(overrides: dict[str, bool]) -> None:
    _FLAG_OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FLAG_OVERRIDES_FILE.write_text(json.dumps(overrides, indent=2))


def _load_ui_settings() -> dict:
    if _UI_SETTINGS_FILE.exists():
        try:
            return yaml.safe_load(_UI_SETTINGS_FILE.read_text()) or {}
        except Exception:
            return {}
    return {}


def _save_ui_settings(data: dict) -> None:
    _UI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _UI_SETTINGS_FILE.write_text(yaml.dump(data, default_flow_style=False))


_LLM_PROVIDER_MAP = {
    "Anthropic":          ("anthropic", None),
    "OpenAI":             ("openai",    "https://api.openai.com/v1"),
    "Ollama":             ("openai",    "http://localhost:11434/v1"),
    "LM Studio":          ("openai",    "http://localhost:1234/v1"),
    "vLLM":               ("openai",    "http://localhost:8000/v1"),
    "OpenAI-compatible":  ("openai",    ""),
}

_LLM_PROVIDER_LABELS = list(_LLM_PROVIDER_MAP.keys())

_LLM_DEFAULT_MODELS = {
    "Anthropic":         "claude-sonnet-4-6",
    "OpenAI":            "gpt-4o",
    "Ollama":            "llama3.1:70b",
    "LM Studio":         "local-model",
    "vLLM":              "meta-llama/Llama-3-8b-instruct",
    "OpenAI-compatible": "gpt-4o",
}


def _apply_ui_settings_to_env(ui_settings: dict) -> None:
    """Push persisted UI LLM settings into os.environ so AppSettings picks them up."""
    provider_label = ui_settings.get("llm_provider_label", "")
    if provider_label in _LLM_PROVIDER_MAP:
        provider_str, default_base = _LLM_PROVIDER_MAP[provider_label]
        os.environ["SMA_LLM_PROVIDER"] = provider_str

        base_url = ui_settings.get("llm_base_url") or default_base or ""
        if base_url:
            os.environ["SMA_LLM_BASE_URL"] = base_url
        elif "SMA_LLM_BASE_URL" in os.environ:
            del os.environ["SMA_LLM_BASE_URL"]

        api_key = ui_settings.get("llm_api_key", "")
        if api_key:
            os.environ["SMA_CLAUDE_API_KEY"] = api_key

    model = ui_settings.get("llm_model", "")
    if model:
        os.environ["SMA_CLAUDE_MODEL"] = model

    neural = ui_settings.get("neural_enrichment_enabled")
    if neural is not None:
        os.environ["SMA_NEURAL_ENRICHMENT_ENABLED"] = "true" if neural else "false"

    for dd_key in ("datadog_start_date", "datadog_end_date", "datadog_log_query",
                   "datadog_hours_lookback", "datadog_page_limit", "datadog_site"):
        val = ui_settings.get(dd_key)
        env_key = f"SMA_{dd_key.upper()}"
        if val is not None and str(val).strip():
            os.environ[env_key] = str(val)
        elif env_key in os.environ:
            del os.environ[env_key]


@st.cache_data(ttl=60, show_spinner=False)
def _run_pipeline() -> dict:
    """Run the observability pipeline and return a serialisable result dict."""
    _apply_ui_settings_to_env(_load_ui_settings())
    settings = AppSettings()
    loop = asyncio.new_event_loop()
    try:
        pipeline = build_pipeline(settings)
        result = loop.run_until_complete(pipeline.run(publish=False))
    finally:
        loop.close()
    return {
        "dashboard": result.dashboard.model_dump(),
        "stories": [s.model_dump() for s in result.stories],
        "findings_count": len(result.findings),
    }


def _flag_label(key: str) -> str:
    return (
        key.split(".")[-1]
        .replace("_", " ")
        .replace("aws", "AWS")
        .replace("gcp", "GCP")
        .replace("api", "API")
        .title()
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙ Settings")
        st.caption("Changes take effect on the next scan.")

        # ── Feature Flags ──
        with st.expander("Feature Flags", expanded=False):
            overrides = _load_flag_overrides()
            settings = AppSettings()
            changed = False

            from scrum_master_assistant.core.feature_flags import build_feature_flag_provider
            provider = build_feature_flag_provider(settings)

            for group, keys in _FLAG_GROUPS.items():
                st.markdown(f"**{group}**")
                for key in keys:
                    current = overrides.get(key, provider.is_enabled(key, default=False))
                    new_val = st.toggle(_flag_label(key), value=current, key=f"flag_{key}")
                    if new_val != current:
                        overrides[key] = new_val
                        changed = True

            if changed:
                _save_flag_overrides(overrides)
                st.success("Flags saved.")
                _run_pipeline.clear()

        # ── Data Source Settings ──
        with st.expander("Data Source Settings", expanded=False):
            import datetime as dt
            ui_settings = _load_ui_settings()
            app_defaults = AppSettings()

            with st.form("source_settings_form"):
                updates: dict = {}

                # ── Datadog (special: includes date/time range) ──
                st.markdown("**Datadog**")
                updates["datadog_site"] = st.text_input(
                    "Site",
                    value=str(ui_settings.get("datadog_site") or app_defaults.datadog_site or ""),
                    key="src_datadog_site",
                )
                updates["datadog_log_query"] = st.text_input(
                    "Log Query",
                    value=str(ui_settings.get("datadog_log_query") or app_defaults.datadog_log_query or "*"),
                    key="src_datadog_log_query",
                )
                col_lh, col_pl = st.columns(2)
                with col_lh:
                    updates["datadog_hours_lookback"] = st.number_input(
                        "Lookback Hours",
                        value=int(ui_settings.get("datadog_hours_lookback") or app_defaults.datadog_hours_lookback or 4),
                        step=1, min_value=1, key="src_datadog_hours_lookback",
                    )
                with col_pl:
                    updates["datadog_page_limit"] = st.number_input(
                        "Page Limit",
                        value=int(ui_settings.get("datadog_page_limit") or app_defaults.datadog_page_limit or 1000),
                        step=100, min_value=100, max_value=1000, key="src_datadog_page_limit",
                    )

                st.caption("Date range overrides Lookback Hours when set.")
                _now_utc = dt.datetime.now(dt.timezone.utc)
                _4h_ago = _now_utc - dt.timedelta(hours=4)

                raw_start = ui_settings.get("datadog_start_date") or ""
                raw_end   = ui_settings.get("datadog_end_date") or ""

                col_s, col_st, col_e, col_et = st.columns(4)
                with col_s:
                    start_date = st.date_input(
                        "Start date",
                        value=dt.date.fromisoformat(raw_start[:10]) if raw_start else _4h_ago.date(),
                        key="dd_start_date",
                    )
                with col_st:
                    start_time = st.time_input(
                        "Start time (UTC)",
                        value=dt.time.fromisoformat(raw_start[11:19]) if len(raw_start) > 10 else _4h_ago.time().replace(second=0, microsecond=0),
                        key="dd_start_time",
                    )
                with col_e:
                    end_date = st.date_input(
                        "End date",
                        value=dt.date.fromisoformat(raw_end[:10]) if raw_end else _now_utc.date(),
                        key="dd_end_date",
                    )
                with col_et:
                    end_time = st.time_input(
                        "End time (UTC)",
                        value=dt.time.fromisoformat(raw_end[11:19]) if len(raw_end) > 10 else _now_utc.time().replace(second=0, microsecond=0),
                        key="dd_end_time",
                    )

                use_range = st.checkbox(
                    "Use date range (uncheck to use Lookback Hours)",
                    value=bool(raw_start or raw_end),
                    key="dd_use_range",
                )
                if use_range:
                    updates["datadog_start_date"] = dt.datetime.combine(start_date, start_time).isoformat()
                    updates["datadog_end_date"]   = dt.datetime.combine(end_date, end_time).isoformat()
                else:
                    updates["datadog_start_date"] = ""
                    updates["datadog_end_date"]   = ""

                st.divider()

                # ── Remaining sources (generic) ──
                _OTHER_SOURCE_SETTINGS = {k: v for k, v in _SOURCE_SETTINGS.items() if k != "Datadog"}
                for source_name, fields in _OTHER_SOURCE_SETTINGS.items():
                    st.markdown(f"**{source_name}**")
                    for field_key, field_label in fields.items():
                        current_val = ui_settings.get(field_key, getattr(app_defaults, field_key, ""))
                        if isinstance(current_val, int):
                            val = st.number_input(field_label, value=current_val, key=f"src_{field_key}", step=1)
                        else:
                            val = st.text_input(field_label, value=str(current_val or ""), key=f"src_{field_key}")
                        updates[field_key] = val

                if st.form_submit_button("Save Source Settings"):
                    ui_settings.update(updates)
                    _save_ui_settings(ui_settings)
                    _run_pipeline.clear()
                    st.success("Saved. Next scan will use updated settings.")

        # ── LLM Configuration ──
        with st.expander("LLM Configuration", expanded=False):
            ui_settings = _load_ui_settings()

            with st.form("llm_config_form"):
                neural_on = st.toggle(
                    "Neural enrichment enabled",
                    value=ui_settings.get(
                        "neural_enrichment_enabled",
                        AppSettings().neural_enrichment_enabled,
                    ),
                )

                current_label = ui_settings.get("llm_provider_label", "Anthropic")
                if current_label not in _LLM_PROVIDER_LABELS:
                    current_label = "Anthropic"
                provider_label = st.selectbox(
                    "Provider",
                    _LLM_PROVIDER_LABELS,
                    index=_LLM_PROVIDER_LABELS.index(current_label),
                    help=(
                        "Anthropic — Anthropic Claude API\n"
                        "OpenAI — OpenAI or any compatible cloud\n"
                        "Ollama / LM Studio / vLLM — local inference servers\n"
                        "OpenAI-compatible — custom endpoint"
                    ),
                )

                _, default_base = _LLM_PROVIDER_MAP[provider_label]

                model_val = st.text_input(
                    "Model name",
                    value=ui_settings.get("llm_model") or _LLM_DEFAULT_MODELS.get(provider_label, ""),
                    help="e.g. claude-sonnet-4-6 / gpt-4o / llama3.1:70b",
                )

                api_key_val = st.text_input(
                    "API key",
                    value=ui_settings.get("llm_api_key", ""),
                    type="password",
                    help="Anthropic / OpenAI key. Leave blank for local models or if set in .env.",
                )

                base_url_val = st.text_input(
                    "Base URL",
                    value=ui_settings.get("llm_base_url") or default_base or "",
                    help="OpenAI-compatible endpoint. e.g. http://localhost:11434/v1 for Ollama.",
                )

                if st.form_submit_button("Save LLM Config"):
                    ui_settings["neural_enrichment_enabled"] = neural_on
                    ui_settings["llm_provider_label"] = provider_label
                    ui_settings["llm_model"] = model_val
                    ui_settings["llm_api_key"] = api_key_val
                    ui_settings["llm_base_url"] = base_url_val
                    _save_ui_settings(ui_settings)
                    _run_pipeline.clear()
                    st.success("LLM config saved. Next scan will use the new settings.")


# ── main pages ────────────────────────────────────────────────────────────────

def _render_overview(data: dict) -> None:
    import pandas as pd

    dash = data["dashboard"]

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Findings", dash["total_findings"])
    c2.metric("Stories", dash["total_stories"])
    security = (
        dash["issues_by_category"].get("pii", 0)
        + dash["issues_by_category"].get("secret", 0)
        + dash["issues_by_category"].get("unauthorized_access", 0)
    )
    c3.metric("Security Issues", security, delta_color="inverse")
    c4.metric("High Confidence", dash["issues_by_confidence_band"].get("high", 0))
    c5.metric("Very High Confidence", dash["issues_by_confidence_band"].get("very-high", 0))

    st.divider()

    row1_l, row1_r = st.columns(2)

    with row1_l:
        st.subheader("Category Mix")
        if dash["issues_by_category"]:
            df = pd.DataFrame.from_dict(dash["issues_by_category"], orient="index", columns=["count"])
            st.bar_chart(df)
        else:
            st.info("No issues detected.")

    with row1_r:
        st.subheader("Severity Breakdown")
        if dash["issues_by_severity"]:
            df = pd.DataFrame.from_dict(dash["issues_by_severity"], orient="index", columns=["count"])
            st.bar_chart(df)
        else:
            st.info("No issues detected.")

    row2_l, row2_r = st.columns(2)

    with row2_l:
        st.subheader("Source Health")
        if dash["source_health"]:
            df = pd.DataFrame.from_dict(dash["source_health"], orient="index", columns=["issues"])
            st.bar_chart(df)
        else:
            st.info("No source data.")

    with row2_r:
        st.subheader("Top Probable Causes")
        if dash["top_probable_causes"]:
            for cause, count in list(dash["top_probable_causes"].items())[:6]:
                st.markdown(f"- **{count}** — {cause}")
        else:
            st.info("No probable causes.")


def _render_stories(data: dict) -> None:
    import pandas as pd

    stories = data["stories"]
    if not stories:
        st.info("No stories generated. Run a scan to populate.")
        return

    rows = []
    for s in stories:
        rows.append({
            "Priority": s["priority"],
            "Epic": s["epic_key"],
            "Summary": s["summary"],
            "Category": s["metadata"].get("category", ""),
            "Sub-category": s["metadata"].get("sub_category", ""),
            "Confidence": round(s["metadata"].get("confidence") or 0, 2),
            "Probable Cause": (s["metadata"].get("probable_cause") or "")[:120],
        })

    df = pd.DataFrame(rows)

    # Filter controls
    col_a, col_b = st.columns(2)
    with col_a:
        priorities = ["All"] + sorted(df["Priority"].unique().tolist())
        sel_priority = st.selectbox("Priority", priorities)
    with col_b:
        categories = ["All"] + sorted(df["Category"].unique().tolist())
        sel_category = st.selectbox("Category", categories)

    if sel_priority != "All":
        df = df[df["Priority"] == sel_priority]
    if sel_category != "All":
        df = df[df["Category"] == sel_category]

    st.caption(f"{len(df)} stories")
    st.dataframe(df, width="stretch", hide_index=True)


def _render_llmops() -> None:
    import pandas as pd

    tracker = get_tracker()
    stats = tracker.stats()
    calls = tracker.recent(limit=50)

    ui_cfg = _load_ui_settings()
    _apply_ui_settings_to_env(ui_cfg)
    settings = AppSettings()
    active = settings.neural_enrichment_enabled and bool(settings.claude_api_key or settings.llm_base_url)

    if not active:
        st.warning(
            "Neural enrichment is not active. "
            "Configure the **LLM Configuration** panel in the sidebar to enable enrichment."
        )

    provider_label = ui_cfg.get("llm_provider_label") or settings.llm_provider
    st.subheader("LLM Telemetry")
    st.caption(
        f"Provider: **{provider_label}** · Model: **{settings.claude_model}**"
        + (f" · Base URL: `{settings.llm_base_url}`" if settings.llm_base_url else "")
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Calls", stats["total_calls"])
    c2.metric("Success Rate", f"{(stats['success_rate'] or 0) * 100:.1f}%" if stats["success_rate"] is not None else "—")
    c3.metric("Total Tokens", f"{stats['total_tokens']:,}" if stats["total_tokens"] else "—")
    c4.metric("Est. Cost", f"${stats['estimated_cost_usd']:.4f}" if stats["estimated_cost_usd"] else "—")
    c5.metric("Avg Latency", f"{stats['avg_latency_ms']}ms" if stats["avg_latency_ms"] else "—")
    c6.metric("p95 Latency", f"{stats['p95_latency_ms']}ms" if stats["p95_latency_ms"] else "—")

    if stats["calls_by_model"]:
        st.divider()
        col_m, col_v = st.columns(2)
        with col_m:
            st.markdown("**Calls by Model**")
            for model, count in stats["calls_by_model"].items():
                st.markdown(f"- `{model}`: {count}")
        with col_v:
            st.markdown("**Calls by Prompt Version**")
            for version, count in stats["calls_by_prompt_version"].items():
                st.markdown(f"- `{version}`: {count}")

    st.divider()
    st.subheader("Recent Calls")

    if not calls:
        st.info("No LLM calls recorded in this session.")
        return

    rows = []
    for c in calls:
        rows.append({
            "ID": c["call_id"],
            "Issue": c["issue_title"][:60],
            "Model": c["model"],
            "Prompt": c["prompt_version"],
            "In Tok": c["input_tokens"] or "—",
            "Out Tok": c["output_tokens"] or "—",
            "Latency ms": c["latency_ms"] if c["latency_ms"] > 0 else "—",
            "Cost $": f"{c['cost_usd']:.5f}" if c["cost_usd"] > 0 else "—",
            "Status": "✅" if c["success"] else f"❌ {c['error'] or ''}",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── routing rules ─────────────────────────────────────────────────────────────

def _get_graph_client():
    settings = AppSettings()
    if not settings.neo4j_enabled or not settings.neo4j_password:
        return None
    from scrum_master_assistant.graph.client import Neo4jGraphClient
    return Neo4jGraphClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )


def _render_routing(client) -> None:
    from scrum_master_assistant.graph.seed import seed_graph
    from scrum_master_assistant.models.config import AppSettings as _S

    if client is None:
        st.warning("Neo4j is not enabled. Set `neo4j_enabled: true` in `config/application.yaml`.")
        return

    settings = _S()

    # ── Seed button ──
    if st.button("Seed defaults (teams + example services + category rules)"):
        try:
            seed_graph(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
            st.success("Graph seeded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Seed failed: {exc}")

    st.divider()

    # ── Teams ──
    st.subheader("Teams")
    teams = client.get_teams()
    team_names = [t["name"] for t in teams]

    if teams:
        import pandas as pd
        st.dataframe(pd.DataFrame(teams), width="stretch", hide_index=True)

    with st.expander("Add / Edit Team"):
        with st.form("team_form"):
            t_name    = st.text_input("Team name (unique key)", key="tf_name")
            t_project = st.text_input("Jira project key", key="tf_project")
            t_epic    = st.text_input("Default epic key", key="tf_epic")
            t_slack   = st.text_input("Slack channel (optional)", key="tf_slack")
            col_save, col_del = st.columns(2)
            with col_save:
                if st.form_submit_button("Save Team"):
                    if t_name and t_project and t_epic:
                        client.upsert_team(t_name, t_project, t_epic, t_slack)
                        st.success(f"Team '{t_name}' saved.")
                        st.rerun()
                    else:
                        st.error("Name, Jira project, and epic are required.")
            with col_del:
                if st.form_submit_button("Delete Team", type="secondary"):
                    if t_name:
                        client.delete_team(t_name)
                        st.success(f"Team '{t_name}' deleted.")
                        st.rerun()

    st.divider()

    # ── Services ──
    st.subheader("Services & Ownership")
    services = client.get_services()

    if services:
        import pandas as pd
        st.dataframe(pd.DataFrame(services), width="stretch", hide_index=True)

    with st.expander("Add / Edit Service"):
        with st.form("service_form"):
            s_name = st.text_input("Service name (as it appears in logs)", key="sf_name")
            s_team = st.selectbox("Owning team", [""] + team_names, key="sf_team")
            col_save, col_del = st.columns(2)
            with col_save:
                if st.form_submit_button("Save Service"):
                    if s_name:
                        client.upsert_service(s_name, s_team)
                        st.success(f"Service '{s_name}' saved.")
                        st.rerun()
                    else:
                        st.error("Service name is required.")
            with col_del:
                if st.form_submit_button("Delete Service", type="secondary"):
                    if s_name:
                        client.delete_service(s_name)
                        st.success(f"Service '{s_name}' deleted.")
                        st.rerun()

    st.divider()

    # ── Category routing ──
    st.subheader("Category Routing Rules")
    st.caption(
        "Primary team gets a Jira story for every issue in this category. "
        "Enable 'Also notify owner' to fan-out a second story to the service's owning team."
    )

    _CATEGORIES = [
        "security", "pii", "secret", "certificate", "unauthorized_access",
        "api_gateway", "application_error", "resource_exhaustion", "database",
        "service_reachability", "network_issue", "infrastructure", "latency",
        "system_failure", "api", "compliance", "spark_error", "ios_error", "android_error",
    ]

    routings = client.get_category_routings()
    if routings:
        import pandas as pd
        st.dataframe(pd.DataFrame(routings), width="stretch", hide_index=True)

    with st.expander("Add / Edit Category Rule"):
        with st.form("category_form"):
            existing_cats = [r["category"] for r in routings]
            all_cats = sorted(set(_CATEGORIES + existing_cats))
            c_cat    = st.selectbox("Category", all_cats, key="cf_cat")
            c_team   = st.selectbox("Primary team", [""] + team_names, key="cf_team")
            c_notify = st.checkbox("Also notify service owner", key="cf_notify")
            col_save, col_del = st.columns(2)
            with col_save:
                if st.form_submit_button("Save Rule"):
                    if c_cat and c_team:
                        client.upsert_category_routing(c_cat, c_team, c_notify)
                        st.success(f"Rule for '{c_cat}' saved.")
                        st.rerun()
                    else:
                        st.error("Category and primary team are required.")
            with col_del:
                if st.form_submit_button("Delete Rule", type="secondary"):
                    client.delete_category_routing(c_cat)
                    st.success(f"Rule for '{c_cat}' deleted.")
                    st.rerun()


# ── app entry point ───────────────────────────────────────────────────────────

def main() -> None:
    _render_sidebar()

    st.title("🔭 Scrum Master Assistant")
    st.caption("Neuro-symbolic observability control plane — local Python UI")

    col_run, col_clear, _ = st.columns([1, 1, 6])
    with col_run:
        if st.button("▶ Run Scan", type="primary"):
            st.session_state["scan_requested"] = True
            _run_pipeline.clear()
    with col_clear:
        if st.button("Clear Cache"):
            st.session_state.pop("scan_requested", None)
            _run_pipeline.clear()
            st.rerun()

    graph_client = _get_graph_client()

    if not st.session_state.get("scan_requested"):
        st.info("Click **▶ Run Scan** to fetch logs and generate stories.")
        tab_overview, tab_stories, tab_llmops, tab_routing = st.tabs(
            ["Overview", "Stories", "LLMOps", "Routing Rules"]
        )
        with tab_overview:
            st.info("No data yet. Run a scan to populate.")
        with tab_stories:
            st.info("No stories yet. Run a scan to populate.")
        with tab_llmops:
            _render_llmops()
        with tab_routing:
            _render_routing(graph_client)
        return

    with st.spinner("Running observability pipeline…"):
        try:
            data = _run_pipeline()
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    tab_overview, tab_stories, tab_llmops, tab_routing = st.tabs(
        ["Overview", "Stories", "LLMOps", "Routing Rules"]
    )

    with tab_overview:
        _render_overview(data)

    with tab_stories:
        _render_stories(data)

    with tab_llmops:
        _render_llmops()

    with tab_routing:
        _render_routing(graph_client)


if __name__ == "__main__":
    main()
