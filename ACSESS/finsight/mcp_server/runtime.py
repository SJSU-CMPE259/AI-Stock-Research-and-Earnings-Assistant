"""Process-wide setup for the MCP server: paths, .env, mode, and the one Orchestrator.

bootstrap() must run before any backend.app import: backend/app/settings.py reads
.env and ./data relative to the working directory, at import time. Claude Desktop
starts the server with cwd=/, so we chdir to the finsight root first.

Modes (FINSIGHT_FAKE in .env or the environment):
  1 = FakeLLM + keyword routing (no GPU; used by tests and CI)
  0 = MistralLLM (tunnel on :8000) + LayaRouter (LAYA_URL, falls back to keywords)
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # ACSESS/finsight

logger = logging.getLogger("finsight.mcp")

# One Orchestrator per process so follow-up questions keep their conversation.
# Orchestrator.ask() mutates shared memory, so callers hold LOCK around it.
# _build_lock is separate because LOCK isn't reentrant and callers may hold it.
LOCK = threading.Lock()
_build_lock = threading.Lock()
_orchestrator = None


def bootstrap() -> None:
    """chdir to the finsight root, make it importable, load its .env (never overrides env vars)."""
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    load_dotenv(ROOT / ".env")


def fake_mode() -> bool:
    return os.getenv("FINSIGHT_FAKE", "0").strip().lower() in ("1", "true", "yes")


def orchestrator():
    """The process's single Orchestrator, built on first use for the current mode."""
    global _orchestrator
    with _build_lock:
        if _orchestrator is None:
            from agent_core.orchestrator import Orchestrator

            if fake_mode():
                from agent_core.llm import FakeLLM

                _orchestrator = Orchestrator(FakeLLM())
            else:
                from agent_core.laya_router import LayaRouter
                from agent_core.llm import MistralLLM

                _orchestrator = Orchestrator(MistralLLM(), router=LayaRouter())
    return _orchestrator


def _port_open(url: str) -> bool:
    """True if something accepts TCP connections at the URL's host:port."""
    u = urlparse(url)
    try:
        port = u.port or (443 if u.scheme == "https" else 80)  # .port raises ValueError if malformed
        with socket.create_connection((u.hostname or "localhost", port), timeout=0.3):
            return True
    except (OSError, ValueError):
        return False


def startup_report() -> dict:
    """Mode, whether keys are set (booleans only, never values), and endpoint reachability."""
    return {
        "mode": "fake" if fake_mode() else "real",
        "finnhub_key_set": bool(os.getenv("FINNHUB_API_KEY")),
        "sec_user_agent_set": bool(os.getenv("SEC_USER_AGENT")),
        "llm_up": _port_open(os.getenv("REMOTE_LLM_URL", "http://localhost:8000/v1")),
        "laya_up": _port_open(os.getenv("LAYA_URL", "http://localhost:8001")),
    }


def log_startup() -> None:
    logger.info("finsight mcp: " + " ".join(f"{k}={v}" for k, v in startup_report().items()))
