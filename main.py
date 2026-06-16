#!/usr/bin/env python3
"""
Scrum Master Assistant — unified launcher.

Usage
-----
  python main.py --nogui              # headless continuous worker (no UI)
  python main.py --localgui           # Streamlit local Python dashboard
  python main.py --react              # FastAPI backend (pair with React frontend)

Options
-------
  --host HOST       API / Streamlit bind address (default: 127.0.0.1)
  --port PORT       Port for the server (default: 8000; Streamlit uses 8501)
  --reload          Enable hot-reload for FastAPI (dev mode)
  --publish         Publish Jira stories during scans (--nogui only)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Scrum Master Assistant — unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--nogui",
        action="store_true",
        help="Run headless continuous worker (no user interface)",
    )
    mode.add_argument(
        "--localgui",
        action="store_true",
        help="Launch Streamlit local Python dashboard",
    )
    mode.add_argument(
        "--react",
        action="store_true",
        help="Launch FastAPI backend for the React frontend",
    )

    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (FastAPI dev mode)")
    parser.add_argument("--publish", action="store_true", help="Publish Jira stories (--nogui only)")

    return parser.parse_args()


def _ensure_src_on_path() -> None:
    src = str(Path(__file__).parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# ── modes ─────────────────────────────────────────────────────────────────────

def run_nogui(args: argparse.Namespace) -> None:
    """Headless mode: start the continuous scan worker with no UI."""
    _ensure_src_on_path()
    import asyncio
    from scrum_master_assistant.models.config import AppSettings
    from scrum_master_assistant.runtime.factory import build_pipeline, build_queue_backend
    from scrum_master_assistant.runtime.jobs import ScanJob

    print("Starting headless continuous worker (no UI).")
    print("Set SMA_JIRA_PUBLISH_ENABLED=true to publish stories to Jira.\n")

    async def _run() -> None:
        settings = AppSettings()
        if args.publish:
            settings = settings.model_copy(update={"jira_publish_on_scan": True, "jira_publish_enabled": True})
        from scrum_master_assistant.workers.continuous import ContinuousWorker
        worker = ContinuousWorker(settings)
        await worker.run_forever()

    asyncio.run(_run())


def run_localgui(args: argparse.Namespace) -> None:
    """Streamlit mode: launch the local Python dashboard."""
    _ensure_src_on_path()
    app_path = Path(__file__).parent / "streamlit_app.py"
    if not app_path.exists():
        print(f"ERROR: streamlit_app.py not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Streamlit has its own default port (8501). Honour --port if supplied.
    streamlit_port = args.port if args.port != 8000 else 8501

    print(f"Starting Streamlit dashboard on http://{args.host}:{streamlit_port}")
    print("Press Ctrl-C to stop.\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", args.host,
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(
            "ERROR: streamlit is not installed.\n"
            "Install it with:  pip install streamlit\n"
            "  or:             pip install 'scrum-master-assistant[localgui]'",
            file=sys.stderr,
        )
        sys.exit(1)


def run_react(args: argparse.Namespace) -> None:
    """React mode: start the FastAPI backend that the React frontend talks to."""
    _ensure_src_on_path()
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"Starting FastAPI backend on http://{args.host}:{args.port}")
    print(
        f"\nReact frontend:\n"
        f"  cd frontend && npm install && npm run dev\n"
        f"  (Set VITE_API_BASE_URL=http://{args.host}:{args.port} in frontend/.env)\n"
    )
    print("Press Ctrl-C to stop.\n")

    uvicorn.run(
        "scrum_master_assistant.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.nogui:
        run_nogui(args)
    elif args.localgui:
        run_localgui(args)
    elif args.react:
        run_react(args)


if __name__ == "__main__":
    main()
