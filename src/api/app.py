from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scrum_master_assistant.api.security import build_auth_context, build_auth_dependency
from scrum_master_assistant.models.auth import AuthenticatedUser, Role
from scrum_master_assistant.models.config import AppSettings
from scrum_master_assistant.models.issues import IssueFeedback
from scrum_master_assistant.runtime.factory import build_pipeline as build_runtime_pipeline
from scrum_master_assistant.runtime.factory import build_graph_client, build_queue_backend, build_store
from scrum_master_assistant.runtime.jobs import ScanJob

settings = AppSettings()
app = FastAPI(title=settings.app_name, version="0.1.0")
get_current_user = build_auth_dependency(settings)

_FLAG_OVERRIDES_FILE = Path("config/ui-flag-overrides.json")
_UI_SETTINGS_FILE = Path("config/ui-settings.yml")

_FLAG_KEYS: list[str] = [
    "source.datadog",
    "source.aws_cloudwatch",
    "source.aws_cloudtrail",
    "source.azure_monitor",
    "source.gcp_observability",
    "source.splunk",
    "source.elasticsearch",
    "source.aws_opensearch",
    "source.ios_crashlytics",
    "source.android_crashlytics",
    "dashboard.security",
    "dashboard.api",
    "dashboard.infrastructure",
    "dashboard.certificates",
    "dashboard.mobile",
    "system.enable_mock_data",
]

_SYSTEM_FLAGS = {"system.enable_mock_data"}

_SOURCE_SETTINGS_FIELDS: dict[str, list[str]] = {
    "datadog": ["datadog_site", "datadog_log_query", "datadog_hours_lookback", "datadog_page_limit", "datadog_max_pages"],
    "aws_cloudwatch": ["cloudwatch_log_groups", "cloudwatch_hours_lookback"],
    "aws_cloudtrail": ["cloudtrail_hours_lookback"],
    "azure_monitor": ["azure_log_analytics_workspace_ids", "azure_monitor_query", "azure_monitor_hours_lookback"],
    "gcp": ["gcp_project_ids", "gcp_logging_filter", "gcp_logging_hours_lookback"],
    "splunk": ["splunk_base_url", "splunk_search_query", "splunk_hours_lookback", "splunk_max_results", "splunk_verify_ssl"],
    "elasticsearch": ["elasticsearch_url", "elasticsearch_index", "elasticsearch_query", "elasticsearch_hours_lookback", "elasticsearch_max_results", "elasticsearch_verify_ssl"],
    "aws_opensearch": ["opensearch_url", "opensearch_region", "opensearch_index", "opensearch_query", "opensearch_hours_lookback", "opensearch_max_results"],
    "ios_crashlytics": ["firebase_project_id", "firebase_ios_app_ids", "firebase_crash_lookback_days"],
    "android_crashlytics": ["firebase_project_id", "firebase_android_app_ids", "firebase_crash_lookback_days"],
}

_AUTH_STATUS_FIELDS: dict[str, list[str]] = {
    "datadog": ["datadog_api_key", "datadog_app_key"],
    "aws_cloudwatch": ["aws_access_key_id", "aws_role_arn"],
    "aws_cloudtrail": ["aws_access_key_id", "aws_role_arn"],
    "azure_monitor": ["azure_client_id", "azure_client_secret"],
    "gcp": ["gcp_service_account_json"],
    "splunk": ["splunk_token", "splunk_username"],
    "elasticsearch": ["elasticsearch_api_key", "elasticsearch_username"],
    "aws_opensearch": ["aws_access_key_id", "aws_role_arn"],
    "ios_crashlytics": ["firebase_service_account_json"],
    "android_crashlytics": ["firebase_service_account_json"],
}


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


def _apply_ui_settings(base: AppSettings) -> AppSettings:
    """Overlay config/ui-settings.yml onto AppSettings so UI changes take effect immediately."""
    if not _UI_SETTINGS_FILE.exists():
        return base
    try:
        ui = yaml.safe_load(_UI_SETTINGS_FILE.read_text()) or {}
    except Exception:
        return base
    overrides: dict[str, Any] = {}

    # LLM settings
    if "neural_enrichment_enabled" in ui:
        overrides["neural_enrichment_enabled"] = ui["neural_enrichment_enabled"]
    if "llm_model" in ui and ui["llm_model"]:
        overrides["claude_model"] = ui["llm_model"]
    if "llm_provider_label" in ui:
        overrides["llm_provider"] = "anthropic" if ui["llm_provider_label"] == "Anthropic" else "openai"
    if "llm_api_key" in ui and ui["llm_api_key"]:
        overrides["claude_api_key"] = ui["llm_api_key"]
    if "llm_base_url" in ui and ui["llm_base_url"]:
        overrides["llm_base_url"] = ui["llm_base_url"]

    # System flags stored in ui-settings.yml
    if "enable_mock_data" in ui:
        overrides["enable_mock_data"] = ui["enable_mock_data"]

    # Source settings (datadog, cloudwatch, azure, gcp, etc.)
    _source_fields: set[str] = {f for fields in _SOURCE_SETTINGS_FIELDS.values() for f in fields}
    for field in _source_fields:
        if field in ui and ui[field] is not None:
            overrides[field] = ui[field]

    return base.model_copy(update=overrides) if overrides else base


def build_pipeline():
    return build_runtime_pipeline(_apply_ui_settings(AppSettings()))


def get_settings() -> AppSettings:
    return _apply_ui_settings(AppSettings())


def _require_roles(user: AuthenticatedUser, allowed: set[Role], detail: str) -> None:
    if not user.roles.intersection(allowed):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


async def _run_scan(settings: AppSettings, publish: bool) -> dict[str, Any]:
    if settings.temporal_enabled:
        try:
            from temporalio.client import Client  # type: ignore[import]

            from scrum_master_assistant.temporal.workflows import ObservabilityScanWorkflow

            workflow_id = f"observability-scan-{settings.environment}-{uuid4().hex[:12]}"
            client = await Client.connect(
                settings.temporal_target_host,
                namespace=settings.temporal_namespace,
            )
            result = await client.execute_workflow(
                ObservabilityScanWorkflow.run,
                publish,
                id=workflow_id,
                task_queue=settings.temporal_task_queue,
            )
            if isinstance(result, dict):
                result["execution_mode"] = "temporal"
                result["workflow_id"] = workflow_id
                # Replay LLM calls tracked in temporal-worker into this process's tracker
                if result.get("llm_calls"):
                    from scrum_master_assistant.llmops.tracker import LLMCallRecord, get_tracker
                    from datetime import datetime, timezone
                    tracker = get_tracker()
                    for call in result["llm_calls"]:
                        ts_raw = call.get("timestamp")
                        try:
                            ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
                        except Exception:
                            ts = datetime.now(timezone.utc)
                        tracker.record(LLMCallRecord(
                            call_id=call.get("call_id", ""),
                            prompt_version=call.get("prompt_version", "v1"),
                            model=call.get("model", ""),
                            provider=call.get("provider", ""),
                            issue_title=call.get("issue_title", ""),
                            input_tokens=call.get("input_tokens", 0),
                            output_tokens=call.get("output_tokens", 0),
                            latency_ms=float(call.get("latency_ms", 0)),
                            cost_usd=float(call.get("cost_usd", 0)),
                            success=call.get("success", True),
                            error=call.get("error"),
                            timestamp=ts,
                        ))
                return result
        except Exception as exc:
            fallback_result = await build_runtime_pipeline(settings).run(publish=publish)
            db = fallback_result.get("dashboard")
            return {
                "findings": [finding.model_dump(mode="json") for finding in fallback_result.get("findings", [])],
                "issues": [issue.model_dump() for issue in fallback_result.get("issues", [])],
                "stories": [story.model_dump() for story in fallback_result.get("stories", [])],
                "dashboard": db.model_dump() if db is not None else {},
                "skipped_duplicates": fallback_result.get("skipped_duplicates", 0),
                "execution_mode": "synchronous-fallback",
                "temporal_error": str(exc),
            }

    result = await build_runtime_pipeline(settings).run(publish=publish)
    db = result.get("dashboard")
    return {
        "findings": [finding.model_dump(mode="json") for finding in result.get("findings", [])],
        "issues": [issue.model_dump() for issue in result.get("issues", [])],
        "stories": [story.model_dump() for story in result.get("stories", [])],
        "dashboard": db.model_dump() if db is not None else {},
        "skipped_duplicates": result.get("skipped_duplicates", 0),
        "execution_mode": "synchronous",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": get_settings().environment}


@app.get("/auth/context")
async def auth_context(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    context = build_auth_context(get_settings())
    return {"user": user.model_dump(mode="json"), "auth": context.model_dump()}


@app.get("/runtime/mode")
async def runtime_mode(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    runtime_settings = get_settings()
    return {
        "enable_mock_data": runtime_settings.enable_mock_data,
        "continuous_mode_enabled": runtime_settings.continuous_mode_enabled,
        "queue_backend": runtime_settings.queue_backend,
        "queue_path": runtime_settings.queue_path,
        "temporal_enabled": runtime_settings.temporal_enabled,
        "temporal_task_queue": runtime_settings.temporal_task_queue,
        "temporal_namespace": runtime_settings.temporal_namespace,
        "temporal_target_host": runtime_settings.temporal_target_host,
        "persistence_backend": runtime_settings.persistence_backend,
        "persistence_path": runtime_settings.persistence_path,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard_view(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> str:
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Scrum Master Assistant Dashboard</title>
      <style>
        body {
          margin: 0;
          font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
          background: linear-gradient(180deg, #f5efe3 0%, #efe3d0 100%);
          color: #171411;
        }
        main {
          max-width: 760px;
          margin: 0 auto;
          padding: 48px 20px;
        }
        section {
          border: 1px solid rgba(76, 57, 37, 0.14);
          border-radius: 24px;
          background: rgba(255, 250, 243, 0.9);
          padding: 28px;
          box-shadow: 0 24px 60px rgba(46, 34, 24, 0.12);
        }
        h1 {
          margin: 0 0 12px;
          font-family: "Fraunces", Georgia, serif;
          font-size: clamp(2rem, 5vw, 3.4rem);
          line-height: 0.95;
        }
        p {
          color: #655b4f;
          line-height: 1.6;
        }
        code {
          background: rgba(11, 122, 115, 0.1);
          color: #0b7a73;
          padding: 3px 8px;
          border-radius: 999px;
        }
      </style>
    </head>
    <body>
      <main>
        <section>
          <h1>React dashboard scaffold created.</h1>
          <p>The interactive dashboard now lives in <code>frontend/</code> as a Vite + React app.</p>
          <p>Start the API on port <code>8000</code>, then run the frontend with <code>npm install</code> and <code>npm run dev</code> inside <code>frontend/</code>.</p>
        </section>
      </main>
    </body>
    </html>
    """


@app.get("/dashboard")
async def dashboard(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    result = await build_pipeline().run(publish=False)
    return result.dashboard.model_dump()


@app.get("/stories")
async def stories(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    result = await build_pipeline().run(publish=False)
    return {"stories": [story.model_dump() for story in result.stories], "jira_results": result.jira_results}


@app.post("/scan")
async def scan(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.security_analyst, Role.platform_admin}, "security-analyst or platform-admin role required")
    runtime_settings = get_settings()
    return await _run_scan(
        runtime_settings,
        publish=runtime_settings.jira_publish_on_scan and runtime_settings.jira_publish_enabled,
    )


@app.post("/enqueue-scan")
async def enqueue_scan(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.security_analyst, Role.platform_admin}, "security-analyst or platform-admin role required")
    runtime_settings = get_settings()
    queue = build_queue_backend(runtime_settings)
    job = ScanJob(source="api", publish=runtime_settings.jira_publish_on_scan and runtime_settings.jira_publish_enabled)
    await queue.enqueue(job)
    return {"status": "queued", "job": job.model_dump(mode="json")}


@app.get("/flags")
async def get_flags(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    from scrum_master_assistant.core.feature_flags import build_feature_flag_provider
    runtime_settings = get_settings()
    provider = build_feature_flag_provider(runtime_settings)
    flags = {k: provider.is_enabled(k, default=False) for k in _FLAG_KEYS if k not in _SYSTEM_FLAGS}
    flags["system.enable_mock_data"] = runtime_settings.enable_mock_data
    flags.update(_load_flag_overrides())
    return {"flags": flags, "backend": "launchdarkly" if runtime_settings.launchdarkly_sdk_key else "static"}


class FlagUpdate(BaseModel):
    enabled: bool


@app.patch("/flags/{flag_key:path}")
async def update_flag(
    flag_key: str,
    body: FlagUpdate,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    if flag_key not in _FLAG_KEYS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown flag: {flag_key}")

    runtime_settings = get_settings()

    # System flags are stored in ui-settings.yml, not the feature flag provider
    if flag_key in _SYSTEM_FLAGS:
        existing: dict = {}
        if _UI_SETTINGS_FILE.exists():
            try:
                existing = yaml.safe_load(_UI_SETTINGS_FILE.read_text()) or {}
            except Exception:
                pass
        if flag_key == "system.enable_mock_data":
            existing["enable_mock_data"] = body.enabled
        _UI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _UI_SETTINGS_FILE.write_text(yaml.dump(existing, default_flow_style=False))
        return {"flag": flag_key, "enabled": body.enabled, "backend": "static"}

    if runtime_settings.launchdarkly_management_key:
        url = (
            f"https://app.launchdarkly.com/api/v2/flags"
            f"/{runtime_settings.launchdarkly_project_key}/{flag_key}"
        )
        patch_doc = [{"op": "replace", "path": f"/environments/{runtime_settings.launchdarkly_environment_key}/on", "value": body.enabled}]
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.patch(
                url,
                headers={"Authorization": runtime_settings.launchdarkly_management_key, "Content-Type": "application/json"},
                json=patch_doc,
            )
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=f"LaunchDarkly API error: {resp.text[:200]}")
        return {"flag": flag_key, "enabled": body.enabled, "backend": "launchdarkly"}

    overrides = _load_flag_overrides()
    overrides[flag_key] = body.enabled
    _save_flag_overrides(overrides)
    return {"flag": flag_key, "enabled": body.enabled, "backend": "static"}


@app.get("/settings/sources")
async def get_source_settings(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    runtime_settings = get_settings()
    sources: dict[str, dict[str, Any]] = {}
    for source, fields in _SOURCE_SETTINGS_FIELDS.items():
        data: dict[str, Any] = {f: getattr(runtime_settings, f, None) for f in fields}
        auth_fields = _AUTH_STATUS_FIELDS.get(source, [])
        data["_auth_configured"] = any(bool(getattr(runtime_settings, f, None)) for f in auth_fields)
        sources[source] = data
    return {"sources": sources}


class SourceSettingsUpdate(BaseModel):
    settings: dict[str, Any]


@app.put("/settings/sources")
async def update_source_settings(
    body: SourceSettingsUpdate,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    allowed: set[str] = {f for fields in _SOURCE_SETTINGS_FIELDS.values() for f in fields}
    filtered = {k: v for k, v in body.settings.items() if k in allowed}

    existing: dict[str, Any] = {}
    if _UI_SETTINGS_FILE.exists():
        try:
            existing = yaml.safe_load(_UI_SETTINGS_FILE.read_text()) or {}
        except Exception:
            existing = {}
    existing.update(filtered)
    _UI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _UI_SETTINGS_FILE.write_text(yaml.dump(existing, default_flow_style=False))
    return {"updated": list(filtered.keys())}


@app.get("/settings/llm")
async def get_llm_settings(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    runtime_settings = get_settings()
    existing: dict[str, Any] = {}
    if _UI_SETTINGS_FILE.exists():
        try:
            existing = yaml.safe_load(_UI_SETTINGS_FILE.read_text()) or {}
        except Exception:
            existing = {}
    return {
        "neural_enrichment_enabled": existing.get("neural_enrichment_enabled", runtime_settings.neural_enrichment_enabled),
        "llm_provider_label": existing.get("llm_provider_label", "Anthropic" if runtime_settings.llm_provider == "anthropic" else "OpenAI-compatible"),
        "llm_model": existing.get("llm_model", runtime_settings.claude_model),
        "llm_api_key": existing.get("llm_api_key", ""),
        "llm_base_url": existing.get("llm_base_url", runtime_settings.llm_base_url or ""),
        "api_key_configured": bool(runtime_settings.claude_api_key or runtime_settings.llm_base_url),
    }


class LlmSettingsUpdate(BaseModel):
    neural_enrichment_enabled: bool
    llm_provider_label: str
    llm_model: str
    llm_api_key: str = ""
    llm_base_url: str = ""


@app.put("/settings/llm")
async def update_llm_settings(
    body: LlmSettingsUpdate,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    existing: dict[str, Any] = {}
    if _UI_SETTINGS_FILE.exists():
        try:
            existing = yaml.safe_load(_UI_SETTINGS_FILE.read_text()) or {}
        except Exception:
            existing = {}
    existing.update(body.model_dump())
    _UI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _UI_SETTINGS_FILE.write_text(yaml.dump(existing, default_flow_style=False))
    return {"updated": list(body.model_dump().keys())}


@app.get("/llmops/stats")
async def llmops_stats(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    from scrum_master_assistant.llmops.tracker import get_tracker
    runtime_settings = get_settings()
    stats = get_tracker().stats()
    stats["neural_enrichment_enabled"] = runtime_settings.neural_enrichment_enabled
    stats["llm_provider"] = runtime_settings.llm_provider
    stats["llm_model"] = runtime_settings.claude_model
    stats["llm_base_url"] = runtime_settings.llm_base_url
    stats["api_key_configured"] = bool(runtime_settings.claude_api_key or runtime_settings.llm_base_url)
    return stats


@app.get("/llmops/calls")
async def llmops_calls(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    limit: int = 20,
) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    from scrum_master_assistant.llmops.tracker import get_tracker
    return {"calls": get_tracker().recent(limit=min(limit, 100))}


@app.post("/publish")
async def publish(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher or platform-admin role required")
    runtime_settings = get_settings()
    if not runtime_settings.jira_publish_enabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Jira publishing is disabled")
    result = await build_runtime_pipeline(runtime_settings).run(publish=True)
    return {
        "stories": [story.model_dump() for story in result.stories],
        "jira_results": result.jira_results,
        "skipped_duplicates": result.skipped_duplicates,
        "pending_approvals": [a.model_dump(mode="json") for a in result.pending_approvals],
        "correlated_groups": [g.model_dump(mode="json") for g in result.correlated_groups],
    }


# ── Circuit breaker endpoints ─────────────────────────────────────────────────

@app.get("/llmops/circuit-breaker")
async def circuit_breaker_status(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    from scrum_master_assistant.agents.circuit_breaker import get_llm_breaker
    return get_llm_breaker().stats()


@app.post("/llmops/circuit-breaker/reset")
async def circuit_breaker_reset(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    from scrum_master_assistant.agents.circuit_breaker import get_llm_breaker
    get_llm_breaker().reset()
    return {"status": "reset", "state": "closed"}


# ── Phase 2: DLQ endpoints ────────────────────────────────────────────────────

@app.get("/llmops/dlq")
async def llmops_dlq(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    include_resolved: bool = False,
) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    from scrum_master_assistant.llmops.dlq import get_dlq
    dlq = get_dlq()
    return {
        "records": [r.model_dump(mode="json") for r in dlq.list_records(include_resolved=include_resolved)],
        "stats": dlq.stats(),
    }


@app.post("/llmops/dlq/{dlq_id}/resolve")
async def llmops_dlq_resolve(
    dlq_id: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    from scrum_master_assistant.llmops.dlq import get_dlq
    resolved = get_dlq().resolve(dlq_id)
    if not resolved:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"DLQ record not found: {dlq_id}")
    return {"dlq_id": dlq_id, "status": "resolved"}


# ── Phase 2: Approval endpoints ───────────────────────────────────────────────

@app.get("/approval/pending")
async def get_pending_approvals(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher role required")
    store = build_store(get_settings())
    return {"approvals": [a.model_dump(mode="json") for a in store.get_pending_approvals()]}


class ApprovalDecision(BaseModel):
    reason: str | None = None


@app.post("/approval/{approval_id}/approve")
async def approve_story(
    approval_id: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher role required")
    runtime_settings = get_settings()
    store = build_store(runtime_settings)
    updated = store.update_approval_status(approval_id, "approved", resolved_by=user.email)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Approval not found: {approval_id}")
    if runtime_settings.temporal_enabled:
        try:
            from temporalio.client import Client  # type: ignore[import]
            client = await Client.connect(runtime_settings.temporal_target_host, namespace=runtime_settings.temporal_namespace)
            handle = client.get_workflow_handle(f"approval-{approval_id}")
            await handle.signal("approve", user.email or "unknown")
        except Exception:
            pass
    from scrum_master_assistant.core.audit_log import AuditEntry, get_audit_log
    get_audit_log().write(AuditEntry(
        action="approval.granted",
        actor=user.email or "unknown",
        resource_type="approval",
        resource_id=approval_id,
        outcome="success",
    ))
    return {"approval_id": approval_id, "status": "approved", "approved_by": user.email}


@app.post("/approval/{approval_id}/reject")
async def reject_story(
    approval_id: str,
    body: ApprovalDecision,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher role required")
    runtime_settings = get_settings()
    store = build_store(runtime_settings)
    updated = store.update_approval_status(
        approval_id, "rejected", resolved_by=user.email, reason=body.reason or ""
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Approval not found: {approval_id}")
    if runtime_settings.temporal_enabled:
        try:
            from temporalio.client import Client  # type: ignore[import]
            client = await Client.connect(runtime_settings.temporal_target_host, namespace=runtime_settings.temporal_namespace)
            handle = client.get_workflow_handle(f"approval-{approval_id}")
            await handle.signal("reject", body.reason or "", user.email or "unknown")
        except Exception:
            pass
    from scrum_master_assistant.core.audit_log import AuditEntry, get_audit_log
    get_audit_log().write(AuditEntry(
        action="approval.rejected",
        actor=user.email or "unknown",
        resource_type="approval",
        resource_id=approval_id,
        outcome="success",
        metadata={"reason": body.reason or ""},
    ))
    return {"approval_id": approval_id, "status": "rejected", "rejected_by": user.email}


# ── Phase 3: Feedback / confidence calibration ────────────────────────────────

class FeedbackBody(BaseModel):
    is_false_positive: bool
    sub_category: str = "unknown"
    reason: str | None = None


@app.post("/issues/{fingerprint}/feedback")
async def submit_feedback(
    fingerprint: str,
    body: FeedbackBody,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.security_analyst, Role.platform_admin}, "security-analyst role required")
    store = build_store(get_settings())
    feedback = IssueFeedback(
        fingerprint=fingerprint,
        sub_category=body.sub_category,
        is_false_positive=body.is_false_positive,
        reason=body.reason,
        submitted_by=user.email,
    )
    try:
        store.record_feedback(feedback)
    except NotImplementedError:
        pass
    from scrum_master_assistant.core.audit_log import AuditEntry, get_audit_log
    get_audit_log().write(AuditEntry(
        action="feedback.submitted",
        actor=user.email or "unknown",
        resource_type="feedback",
        resource_id=fingerprint,
        outcome="success",
        metadata={"is_false_positive": body.is_false_positive, "sub_category": body.sub_category},
    ))
    return {
        "feedback_id": feedback.feedback_id,
        "fingerprint": fingerprint,
        "is_false_positive": body.is_false_positive,
    }


# ── Phase 3: Trend analysis ────────────────────────────────────────────────────

@app.get("/trends")
async def trends(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    hours: int = 168,
) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    store = build_store(get_settings())
    try:
        records = store.get_recent_issues(hours=min(hours, 720), limit=2000)
    except NotImplementedError:
        return {"error": "Trend analysis requires json or postgres persistence backend", "records": []}

    from collections import Counter
    neural_enriched = sum(1 for r in records if r.neural_enriched)
    avg_confidence = round(sum(r.confidence for r in records) / len(records), 3) if records else None

    return {
        "lookback_hours": hours,
        "total_issues": len(records),
        "neural_enriched": neural_enriched,
        "avg_confidence": avg_confidence,
        "by_category": dict(Counter(r.category for r in records)),
        "by_sub_category": dict(Counter(r.sub_category or "unknown" for r in records)),
        "by_severity": dict(Counter(r.severity for r in records)),
        "by_source": dict(Counter(r.source_system for r in records)),
        "top_affected_services": dict(Counter(r.service_name or "unknown" for r in records).most_common(10)),
    }


# ── Phase 3: On-demand correlation ────────────────────────────────────────────

@app.post("/correlate")
async def correlate(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    result = await build_pipeline().run(publish=False)
    return {
        "correlated_groups": [g.model_dump(mode="json") for g in result.correlated_groups],
        "issues": len(result.issues),
        "correlations_found": len(result.correlated_groups),
    }


# ── Bidirectional Jira sync ────────────────────────────────────────────────────

class JiraStatusPushBody(BaseModel):
    platform_status: str  # "open" | "in_progress" | "closed"


@app.get("/routing/graph")
async def get_routing_graph(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    runtime_settings = get_settings()
    client = build_graph_client(runtime_settings)
    if client is None:
        return {"enabled": False, "teams": [], "services": [], "routings": []}
    return {
        "enabled": True,
        "teams": client.get_teams(),
        "services": client.get_services(),
        "routings": client.get_category_routings(),
    }


@app.post("/routing/seed")
async def seed_routing_graph(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    runtime_settings = get_settings()
    if not runtime_settings.neo4j_enabled or not runtime_settings.neo4j_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    from scrum_master_assistant.graph.seed import seed_graph

    seed_graph(runtime_settings.neo4j_uri, runtime_settings.neo4j_user, runtime_settings.neo4j_password)
    return {"status": "seeded"}


class TeamBody(BaseModel):
    name: str
    jira_project: str
    default_epic: str
    slack: str = ""


@app.put("/routing/team")
async def upsert_routing_team(
    body: TeamBody,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    client = build_graph_client(get_settings())
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    client.upsert_team(body.name, body.jira_project, body.default_epic, body.slack)
    return {"status": "saved", "team": body.name}


@app.delete("/routing/team/{name}")
async def delete_routing_team(
    name: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    client = build_graph_client(get_settings())
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    client.delete_team(name)
    return {"status": "deleted", "team": name}


class ServiceBody(BaseModel):
    service_name: str
    team_name: str = ""


@app.put("/routing/service")
async def upsert_routing_service(
    body: ServiceBody,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    client = build_graph_client(get_settings())
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    client.upsert_service(body.service_name, body.team_name)
    return {"status": "saved", "service": body.service_name}


@app.delete("/routing/service/{service_name}")
async def delete_routing_service(
    service_name: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    client = build_graph_client(get_settings())
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    client.delete_service(service_name)
    return {"status": "deleted", "service": service_name}


class CategoryRoutingBody(BaseModel):
    category: str
    primary_team: str
    also_notify_owner: bool = False


@app.put("/routing/category")
async def upsert_routing_category(
    body: CategoryRoutingBody,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    client = build_graph_client(get_settings())
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    client.upsert_category_routing(body.category, body.primary_team, body.also_notify_owner)
    return {"status": "saved", "category": body.category}


@app.delete("/routing/category/{category}")
async def delete_routing_category(
    category: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    client = build_graph_client(get_settings())
    if client is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Neo4j is not enabled")
    client.delete_category_routing(category)
    return {"status": "deleted", "category": category}


from fastapi import Request as _FastAPIRequest  # noqa: E402


@app.post("/webhooks/jira")
async def jira_webhook_handler(request: _FastAPIRequest) -> dict:
    """Receive real-time Jira issue_updated webhook and sync status into platform."""
    from scrum_master_assistant.core.credentials import JiraCredentials
    from scrum_master_assistant.core.jira_sync import JiraSyncService
    from scrum_master_assistant.integrations.jira import JiraClient

    body = await request.body()
    # Verify HMAC if a webhook secret is configured
    if settings.jira_webhook_secret:
        sig = request.headers.get("x-hub-signature-256", "")
        if not JiraSyncService.verify_webhook_signature(body, sig, settings.jira_webhook_secret):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    creds = JiraCredentials(
        mode=settings.jira_auth_mode,
        username=settings.jira_username,
        api_token=settings.jira_api_token,
    )
    client = JiraClient(base_url=settings.jira_base_url, credentials=creds)
    svc = JiraSyncService(jira_client=client, store=build_store(settings))
    result = await svc.handle_webhook_event(payload)
    return result


@app.post("/jira/sync")
async def trigger_jira_sync(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    """Manually trigger a full polling sync of all tracked Jira tickets."""
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher role required")

    from scrum_master_assistant.core.credentials import JiraCredentials
    from scrum_master_assistant.core.jira_sync import JiraSyncService
    from scrum_master_assistant.integrations.jira import JiraClient

    creds = JiraCredentials(
        mode=settings.jira_auth_mode,
        username=settings.jira_username,
        api_token=settings.jira_api_token,
    )
    client = JiraClient(base_url=settings.jira_base_url, credentials=creds)
    svc = JiraSyncService(jira_client=client, store=build_store(settings))
    report = await svc.sync_all()
    return report.model_dump(mode="json")


@app.get("/jira/sync/status")
async def jira_sync_status(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    """Return platform_status and jira_status for all tracked stories."""
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    store = build_store(settings)
    stories = store.get_all_published_stories()
    return {
        "total": len(stories),
        "stories": [
            {
                "fingerprint": s.fingerprint,
                "jira_issue_key": s.jira_issue_key,
                "summary": s.summary,
                "jira_status": s.jira_status,
                "platform_status": s.platform_status,
                "last_synced_at": s.last_synced_at.isoformat() if s.last_synced_at else None,
            }
            for s in stories
        ],
    }


@app.post("/jira/stories/{fingerprint}/push-status")
async def push_story_status_to_jira(
    fingerprint: str,
    body: JiraStatusPushBody,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> dict:
    """Transition the Jira issue for a story to match the given platform_status."""
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher role required")

    from scrum_master_assistant.core.credentials import JiraCredentials
    from scrum_master_assistant.core.jira_sync import JiraSyncService
    from scrum_master_assistant.integrations.jira import JiraClient

    creds = JiraCredentials(
        mode=settings.jira_auth_mode,
        username=settings.jira_username,
        api_token=settings.jira_api_token,
    )
    client = JiraClient(base_url=settings.jira_base_url, credentials=creds)
    svc = JiraSyncService(jira_client=client, store=build_store(settings))
    result = await svc.push_platform_status_to_jira(fingerprint, body.platform_status)
    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result)
    return result


# ── AgentOps: Evals ───────────────────────────────────────────────────────────

_last_eval_report: dict | None = None


@app.post("/evals/run")
async def run_evals(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    eval_file: str = "data/evals/eval_cases.yaml",
) -> dict:
    """Run the eval suite against the current pipeline components. Returns a full report."""
    _require_roles(user, {Role.security_analyst, Role.platform_admin}, "security-analyst role required")

    from scrum_master_assistant.agents.hallucination_guard import get_guard
    from scrum_master_assistant.core.audit_log import AuditEntry, get_audit_log
    from scrum_master_assistant.core.pii_masker import get_masker
    from scrum_master_assistant.detection.rules import IssueDetector
    from scrum_master_assistant.evals.runner import EvalRunner

    runner = EvalRunner(
        detector=IssueDetector(),
        guard=get_guard(),
        pii_masker=get_masker(),
    )
    report = runner.run_from_file(eval_file)
    report_dict = report.as_dict()

    global _last_eval_report
    _last_eval_report = report_dict

    get_audit_log().write(AuditEntry(
        action="eval.run",
        actor=user.email or "unknown",
        resource_type="eval",
        resource_id=eval_file,
        outcome="success",
        metadata={
            "pass_rate": report.pass_rate,
            "total": report.total,
            "passed": report.passed,
        },
    ))

    return report_dict


@app.get("/evals/results")
async def eval_results(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> dict:
    """Return the most recent eval run report."""
    _require_roles(user, {Role.viewer, Role.security_analyst, Role.jira_publisher, Role.platform_admin}, "viewer role required")
    if _last_eval_report is None:
        return {"message": "No eval run yet. POST /evals/run to execute.", "report": None}
    return {"report": _last_eval_report}


# ── AgentOps: Audit log ───────────────────────────────────────────────────────

@app.get("/audit")
async def audit_log(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    limit: int = 100,
    from_file: bool = False,
) -> dict:
    """Return recent audit log entries. from_file=true reads the durable JSONL file."""
    _require_roles(user, {Role.platform_admin}, "platform-admin role required")
    from scrum_master_assistant.core.audit_log import get_audit_log
    log = get_audit_log()
    entries = log.read_file(limit=min(limit, 1000)) if from_file else log.recent(limit=min(limit, 500))
    return {"entries": entries, "count": len(entries), "source": "file" if from_file else "memory"}


# ── AgentOps: Compliance report ───────────────────────────────────────────────

@app.get("/reports/compliance")
async def compliance_report(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    hours: int = 24,
) -> dict:
    """
    Generate a compliance summary for the last `hours` hours.

    Covers: scan activity, story publishing, approval decisions, feedback,
    eval pass rates, security events (injection attempts, PII detections).
    """
    _require_roles(user, {Role.jira_publisher, Role.platform_admin}, "jira-publisher role required")

    from scrum_master_assistant.agents.circuit_breaker import get_llm_breaker
    from scrum_master_assistant.core.audit_log import get_audit_log
    from scrum_master_assistant.llmops.tracker import get_tracker

    audit_summary = get_audit_log().compliance_summary(hours=hours)
    llm_stats = get_tracker().stats()
    cb_stats = get_llm_breaker().stats()

    # Eval pass rate from last run
    eval_summary = {"last_run": None}
    if _last_eval_report:
        eval_summary = {
            "last_run": _last_eval_report.get("run_at"),
            "pass_rate": _last_eval_report.get("pass_rate"),
            "total_cases": _last_eval_report.get("total"),
            "passed": _last_eval_report.get("passed"),
            "failed": _last_eval_report.get("failed"),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": hours,
        "audit": audit_summary,
        "llmops": {
            "total_calls": llm_stats.get("total_calls", 0),
            "success_rate": llm_stats.get("success_rate", 0),
            "total_cost_usd": llm_stats.get("total_cost_usd", 0),
            "circuit_breaker": cb_stats,
        },
        "evals": eval_summary,
        "prompt_version": "v1.2",
    }
