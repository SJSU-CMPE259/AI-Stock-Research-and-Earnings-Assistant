"""server.py as a real stdio subprocess, launched the way an MCP client may launch it (cwd=/).

Offline: the calls used here (a refusal and argument errors) never reach Finnhub, SEC or an LLM.
"""
import os
import subprocess
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parents[1] / "server.py"
DUMMY_KEY = "test-key-XYZ-must-not-be-logged"
NO_ADVICE = "Facts only, no investment advice."


def test_server_starts_from_root_cwd_and_exits_cleanly_on_eof():
    env = {**os.environ, "FINSIGHT_FAKE": "1"}
    p = subprocess.run([sys.executable, str(SERVER)], cwd="/", env=env, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=30)
    assert p.stdout == ""  # stdout is the protocol channel
    assert "mode=fake" in p.stderr
    assert p.returncode == 0, p.stderr


def _stdio_session(tmp_path, fn, **env):
    """Run fn(session, init_result) against a stdio subprocess; return (fn's result, its stderr)."""
    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER)], cwd="/",
        env={"FINSIGHT_FAKE": "1", "SEC_USER_AGENT": "FinSight Test test@example.com", **env})
    errlog = tmp_path / "server_stderr.log"

    async def go():
        with errlog.open("w") as err, anyio.fail_after(30):
            async with stdio_client(params, errlog=err) as (read, write):
                async with ClientSession(read, write) as session:
                    return await fn(session, await session.initialize())

    return anyio.run(go), errlog.read_text()


def test_stdio_session_from_root_cwd_lists_and_calls_tools(tmp_path):
    async def session_calls(s, init):
        return init, await s.list_tools(), await s.call_tool("ask_finsight", {"question": "Should I buy AMD?"})

    (init, listed, refusal), stderr = _stdio_session(tmp_path, session_calls)
    assert init.serverInfo.name == "finsight"
    assert NO_ADVICE in init.instructions
    assert len(listed.tools) == 5
    assert not refusal.isError and refusal.structuredContent["route"] == "refuse"
    assert "mode=fake" in stderr


def test_keys_never_logged(tmp_path):
    async def session_calls(s, init):
        return [await s.call_tool("ask_finsight", {"question": "Should I buy AMD?"}),
                await s.call_tool("get_company_news", {"ticker": "NVDA", "days": 0}),
                await s.call_tool("get_filing_section", {"ticker": "NVDA", "form": "8-K", "item": "1"})]

    results, stderr = _stdio_session(tmp_path, session_calls, FINNHUB_API_KEY=DUMMY_KEY)
    assert "finnhub_key_set=True" in stderr  # the server saw the key...
    assert DUMMY_KEY not in stderr           # ...and never wrote it out
    assert [r.isError for r in results] == [False, True, True]
    for r in results:
        assert DUMMY_KEY not in r.model_dump_json()
