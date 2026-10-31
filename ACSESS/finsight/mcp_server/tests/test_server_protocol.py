"""server.py over the MCP protocol: registration, then each tool called through a real
ClientSession on in-memory streams (Finnhub, SEC and the LLM stubbed)."""
import re
import threading

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from agent_core.llm import FakeLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator
from agent_core.tools import finnhub_client, prices
from mcp_server import runtime, server, tools
from mcp_server.tests.test_tools_filings import FakeEdgar

TOOL_NAMES = {"get_quote", "get_company_news", "list_filings", "get_filing_section", "ask_finsight"}
NO_ADVICE = "Facts only, no investment advice."
CITATION = re.compile(r"\[[A-Z]\d+\]")
QUOTE = {"c": 182.4, "d": 2.25, "dp": 1.25, "h": 183.0, "l": 179.5,
         "o": 180.0, "pc": 180.15, "t": 1793390400}


def _list_tools():
    return anyio.run(server.mcp.list_tools)


def test_lists_exactly_the_five_tools():
    assert {t.name for t in _list_tools()} == TOOL_NAMES


def test_every_description_and_the_instructions_say_facts_only():
    for t in _list_tools():
        assert NO_ADVICE in t.description, t.name
    assert NO_ADVICE in server.mcp.instructions


def test_tools_are_read_only_open_world_with_structured_output():
    for t in _list_tools():
        assert t.annotations.readOnlyHint is True, t.name
        assert t.annotations.openWorldHint is True, t.name
        assert t.outputSchema is not None, t.name  # -> dict[str, Any] gives structuredContent


def test_blocking_tool_runs_off_the_event_loop_thread(monkeypatch):
    seen = []

    def fake_get_quote(ticker):
        seen.append(threading.get_ident())
        return {"ticker": ticker}

    monkeypatch.setattr(tools, "get_quote", fake_get_quote)

    async def call():
        loop_thread = threading.get_ident()
        await server.mcp.call_tool("get_quote", {"ticker": "NVDA"})
        return loop_thread

    loop_thread = anyio.run(call)
    assert seen and seen[0] != loop_thread


# ---- over the protocol (ClientSession <-> server on in-memory streams) -------

@pytest.fixture
def offline(monkeypatch):
    """Finnhub returns QUOTE and no news, SEC is a FakeEdgar, and a fresh FakeLLM Orchestrator."""
    monkeypatch.setattr(prices, "_get", lambda params: QUOTE)
    monkeypatch.setattr(finnhub_client, "get", lambda path, params, ttl=0: [])
    monkeypatch.setattr(tools, "_edgar", lambda: FakeEdgar())
    orch = Orchestrator(FakeLLM(), memory=SharedMemory(), conversation=Conversation())
    monkeypatch.setattr(runtime, "_orchestrator", orch)
    prices._cache.clear()
    tools._sections_cache.clear()
    yield orch
    prices._cache.clear()
    tools._sections_cache.clear()


def _session(fn):
    async def go():
        async with create_connected_server_and_client_session(server.mcp) as session:
            return await fn(session)
    return anyio.run(go)


def _call(name, args):
    return _session(lambda s: s.call_tool(name, args))


def test_client_lists_five_tools_with_wording_and_annotations():
    listed = _session(lambda s: s.list_tools()).tools
    assert {t.name for t in listed} == TOOL_NAMES
    for t in listed:
        assert NO_ADVICE in t.description, t.name
        assert t.annotations.readOnlyHint is True, t.name


def test_get_quote_over_protocol_has_q1_citation(offline):
    r = _call("get_quote", {"ticker": "NVDA"})
    assert not r.isError
    out = r.structuredContent
    assert out["ticker"] == "NVDA" and out["live"] is True
    assert CITATION.search(out["citation"]).group() == "[Q1]"


def test_get_company_news_over_protocol_is_labeled_untrusted(offline):
    r = _call("get_company_news", {"ticker": "NVDA", "days": 3})
    assert not r.isError
    out = r.structuredContent
    assert out["content_type"] == "untrusted_external_text" and out["notice"]
    assert out["articles"] == [] and "note" in out


def test_list_filings_over_protocol(offline):
    r = _call("list_filings", {"ticker": "NVDA"})
    assert not r.isError
    assert r.structuredContent["filings"][0]["accession"] == "0001045810-26-000010"


def test_get_filing_section_over_protocol_pages_with_s1_citation(offline):
    r = _call("get_filing_section", {"ticker": "NVDA", "form": "10-K", "item": "risk factors",
                                     "max_chars": 500})
    assert not r.isError
    out = r.structuredContent
    assert out["item"] == "Item 1A" and out["returned_chars"] == 500 and out["next_offset"] == 500
    assert CITATION.search(out["citation"]).group() == "[S1]"


def test_ask_finsight_over_protocol_refuses_without_llm(offline):
    r = _call("ask_finsight", {"question": "Should I buy AMD?"})
    assert not r.isError
    assert r.structuredContent["route"] == "refuse"
    assert offline.llm.calls == []


def test_ask_finsight_over_protocol_keeps_citations_and_live_flag(offline):
    out = _call("ask_finsight", {"question": "What is NVDA's price right now?"}).structuredContent
    assert out["quote_live"] is True and out["data_mode"] == "live"
    assert "[Q1]" in CITATION.findall(out["answer"])


def test_tool_failure_is_an_mcp_error_and_the_session_survives(offline, monkeypatch):
    monkeypatch.setattr(prices, "_get", lambda params: QUOTE if params["symbol"] == "NVDA" else {"c": 0, "t": 0})

    async def bad_then_good(s):
        return await s.call_tool("get_quote", {"ticker": "ZZZZ"}), await s.call_tool("get_quote", {"ticker": "NVDA"})

    bad, good = _session(bad_then_good)
    assert bad.isError and bad.structuredContent is None
    assert "No price for 'ZZZZ'" in bad.content[0].text
    assert not good.isError and good.structuredContent["ticker"] == "NVDA"


@pytest.mark.parametrize("name, args, match", [
    ("get_company_news", {"ticker": "NVDA", "days": 0}, "days must be between"),
    ("get_company_news", {"ticker": "NVDA", "days": "a week"}, "days"),
    ("get_filing_section", {"ticker": "NVDA", "form": "8-K", "item": "Item 1"}, "No section patterns for 8-K"),
    ("ask_finsight", {"question": "  "}, "question is empty"),
    ("get_quote", {}, "ticker"),
])
def test_bad_arguments_are_mcp_errors_with_the_reason(offline, name, args, match):
    r = _call(name, args)
    assert r.isError
    assert match in r.content[0].text
