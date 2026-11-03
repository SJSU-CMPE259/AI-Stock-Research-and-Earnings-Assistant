"""runtime.py: bootstrap from any cwd, one Orchestrator, startup log without key values."""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from agent_core.llm import FakeLLM
from mcp_server import runtime

DUMMY_KEY = "test-key-XYZ"


def test_bootstrap_chdirs_to_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime.bootstrap()
    assert Path.cwd() == runtime.ROOT
    assert (runtime.ROOT / "agent_core").is_dir()


def test_import_from_root_cwd_writes_nothing_to_stdout():
    # Mirrors server.py: put ROOT on sys.path, then import tools. Importing tools must
    # bootstrap on its own, so the cwd-relative caches and settings resolve under ROOT.
    code = (
        "import logging, os, sys\n"
        f"sys.path.insert(0, {str(runtime.ROOT)!r})\n"
        "logging.basicConfig(stream=sys.stderr, level=logging.INFO)\n"
        "from mcp_server import runtime, tools\n"
        "assert os.path.samefile(os.getcwd(), runtime.ROOT), os.getcwd()\n"
        "runtime.log_startup()\n"
        "tools._edgar()\n"
        "runtime.orchestrator()\n"
    )
    # Pin the LLM settings so a developer's .env can't fail the backend settings check.
    env = {**os.environ, "FINSIGHT_FAKE": "1", "FINNHUB_API_KEY": DUMMY_KEY,
           "SEC_USER_AGENT": "FinSight Test test@example.com",
           "LLM_PROVIDER": "remote", "REMOTE_LLM_URL": "http://localhost:8000/v1"}
    proc = subprocess.run([sys.executable, "-c", code], cwd="/", env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert "mode=fake" in proc.stderr
    assert DUMMY_KEY not in proc.stderr


def test_startup_report_has_booleans_not_key_values(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", DUMMY_KEY)
    monkeypatch.setattr(runtime, "_port_open", lambda url: False)
    report = runtime.startup_report()
    assert report["mode"] == "fake"
    assert report["finnhub_key_set"] is True
    assert report["sec_user_agent_set"] is True
    assert report["llm_up"] is False and report["laya_up"] is False
    assert DUMMY_KEY not in repr(report)


def test_fake_mode_gives_one_orchestrator_with_fake_llm_and_keyword_routing(monkeypatch):
    monkeypatch.setattr(runtime, "_orchestrator", None)
    first = runtime.orchestrator()
    assert runtime.orchestrator() is first
    assert isinstance(first.llm, FakeLLM)
    assert first.router is None


def test_concurrent_first_calls_build_one_orchestrator(monkeypatch):
    import agent_core.orchestrator as orch_mod

    built = []

    class SlowOrchestrator:
        def __init__(self, llm, router=None):
            time.sleep(0.05)  # widen the check-then-set window
            built.append(self)

    monkeypatch.setattr(orch_mod, "Orchestrator", SlowOrchestrator)
    monkeypatch.setattr(runtime, "_orchestrator", None)
    results = []
    threads = [threading.Thread(target=lambda: results.append(runtime.orchestrator())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
    assert all(r is built[0] for r in results)


def test_port_probe_defaults_https_to_443_and_survives_bad_port(monkeypatch):
    seen = []

    def fake_connect(addr, timeout):
        seen.append(addr)
        raise OSError("refused")

    monkeypatch.setattr(runtime.socket, "create_connection", fake_connect)
    assert runtime._port_open("https://laya.example") is False
    assert seen == [("laya.example", 443)]
    assert runtime._port_open("http://localhost:notaport") is False
