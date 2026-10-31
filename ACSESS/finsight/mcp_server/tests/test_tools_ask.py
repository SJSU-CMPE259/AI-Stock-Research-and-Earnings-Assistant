"""tools.ask_finsight over a FakeLLM Orchestrator, with Finnhub stubbed (no network)."""
import json
import re

import pytest

from agent_core.llm import FakeLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator
from agent_core.tools import finnhub_client, prices
from mcp_server import runtime, tools

QUOTE = {"c": 182.4, "d": 2.25, "dp": 1.25, "h": 183.0, "l": 179.5,
         "o": 180.0, "pc": 180.15, "t": 1793390400}


@pytest.fixture
def orch(monkeypatch):
    o = Orchestrator(FakeLLM(), memory=SharedMemory(), conversation=Conversation())
    monkeypatch.setattr(runtime, "_orchestrator", o)
    monkeypatch.setattr(prices, "_get", lambda params: QUOTE)
    monkeypatch.setattr(finnhub_client, "get", lambda path, params, ttl=0: [])
    prices._cache.clear()
    yield o
    prices._cache.clear()


def _finnhub_down(params):
    raise RuntimeError("Finnhub rejected the request")


def test_ask_refuses_buy_and_never_calls_llm(orch):
    out = tools.ask_finsight("Should I buy AMD?")
    assert out["route"] == "refuse"
    assert "investment advice" in out["answer"].lower()
    assert orch.llm.calls == []
    assert out["quote_live"] is None and out["data_mode"] is None and out["warnings"] == []


def test_ask_real_time_live_true_with_fake_quote(orch):
    out = tools.ask_finsight("What is NVDA's price right now?")
    assert out["route"] == "real_time" and out["route_source"] == "keyword"
    assert out["ticker"] == "NVDA"
    assert out["quote_live"] is True and out["data_mode"] == "live"
    assert out["warnings"] == []
    assert re.search(r"\[Q1\]", out["answer"])  # citation passed through verbatim
    for key in ("route_confidence", "steps", "reused", "run_id", "elapsed_s"):
        assert key in out
    json.dumps(out)


def test_ask_real_time_sample_when_finnhub_fails(orch, monkeypatch):
    monkeypatch.setattr(prices, "_get", _finnhub_down)
    out = tools.ask_finsight("What is NVDA's price right now?")
    assert out["quote_live"] is False and out["data_mode"] == "sample"
    assert any("live price unavailable" in w for w in out["warnings"])


def test_ask_followup_remembers_ticker(orch):
    assert tools.ask_finsight("What was NVDA's revenue last quarter?")["ticker"] == "NVDA"
    follow = tools.ask_finsight("How did it change from the year before?")
    assert follow["ticker"] == "NVDA"
    assert runtime.orchestrator() is orch


def test_ask_numbers_route_warns_sample_data(orch):
    out = tools.ask_finsight("What was NVDA's revenue last quarter?")
    assert out["route"] == "numbers"
    assert out["quote_live"] is None and out["data_mode"] == "sample"
    assert any("AMD sample data" in w and "not NVDA data" in w for w in out["warnings"])


def test_ask_sample_facts_then_live_quote_is_mixed(orch):
    tools.ask_finsight("What was AMD's revenue last quarter?")
    out = tools.ask_finsight("What is AMD's price right now?")
    assert out["quote_live"] is True
    assert out["data_mode"] == "mixed"  # report cites the live quote and the earlier sample notes
    assert any("AMD sample data" in w for w in out["warnings"])


def test_ask_unknown_ticker_warns(orch):
    out = tools.ask_finsight("What is GOOGL's price right now?")
    assert out["ticker"] is None
    assert any("ticker not recognized" in w and "NVDA" in w for w in out["warnings"])


def test_ask_laya_down_warns_keyword_routing(orch):
    class DownRouter:
        source = "laya"

        def route(self, question):
            raise ConnectionError("Laya endpoint down")

    orch.router = DownRouter()
    out = tools.ask_finsight("What is NVDA's price right now?")
    assert out["route_source"] == "keyword"
    assert any("Laya unavailable" in w for w in out["warnings"])


def test_ask_new_conversation_forgets_ticker_and_data(orch):
    tools.ask_finsight("What was NVDA's revenue last quarter?")
    out = tools.ask_finsight("How did it change from the year before?", new_conversation=True)
    assert out["ticker"] is None
    assert out["reused"] == []
    assert runtime.orchestrator() is orch  # same Orchestrator, fresh memory and conversation


def _prices_by_symbol(monkeypatch):
    """A different price per ticker, so a reused quote shows up as the wrong number."""
    prices_ = {"AMD": 150.5, "NVDA": 182.4}
    monkeypatch.setattr(prices, "_get", lambda params: {**QUOTE, "c": prices_[params["symbol"]]})


def _facts_given_to_llm(orch) -> str:
    """The Facts block of the last report prompt (the earlier answer, quoted above it, may cite [Q1])."""
    return orch.llm.calls[-1]["messages"][0]["content"].split("Facts:\n", 1)[1]


def test_ask_ticker_switch_does_not_reuse_other_quote(orch, monkeypatch):
    _prices_by_symbol(monkeypatch)
    tools.ask_finsight("What is AMD's price right now?")
    out = tools.ask_finsight("What is NVDA's price right now?")
    assert out["ticker"] == "NVDA"
    assert out["reused"] == []
    facts = _facts_given_to_llm(orch)
    assert "[Q1] Current price: 182.4" in facts and "150.5" not in facts


def test_ask_ticker_switch_does_not_cite_other_quote(orch, monkeypatch):
    _prices_by_symbol(monkeypatch)
    tools.ask_finsight("What is AMD's price right now?")
    out = tools.ask_finsight("What was NVDA's revenue last quarter?")
    assert out["route"] == "numbers"
    assert "[Q1]" not in _facts_given_to_llm(orch)  # numbers route has no market agent
    assert out["quote_live"] is None and out["data_mode"] == "sample"


def test_ask_ticker_switch_does_not_reuse_other_filing_data(orch):
    tools.ask_finsight("What was AMD's revenue last quarter?")
    out = tools.ask_finsight("What was NVDA's revenue last quarter?")
    assert out["reused"] == []
    assert not any("reused stored data" in s for s in out["steps"])


def test_ask_same_ticker_followup_still_reuses(orch):
    tools.ask_finsight("What is AMD's price right now?")
    out = tools.ask_finsight("What is its price today?")
    assert out["ticker"] == "AMD"
    assert out["reused"] == ["amd_quote"]


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="K3: keyword router matches substrings (buy/hold/sell); agent_core owns it")
@pytest.mark.parametrize("question", [
    "How much did AMD spend on share buybacks last year?",
    "Who are NVDA's largest shareholders?",
    "What were AMD's selling, general and administrative expenses?",
])
def test_ask_keyword_router_does_not_refuse_non_advice(orch, question):
    assert tools.ask_finsight(question)["route"] != "refuse"


@pytest.mark.parametrize("question", ["", "   ", "x" * (tools.MAX_QUESTION_CHARS + 1)])
def test_ask_rejects_empty_or_huge_question(orch, question):
    with pytest.raises(ValueError, match="question"):
        tools.ask_finsight(question)
    assert orch.llm.calls == []


def test_ask_runs_under_the_lock(orch, monkeypatch):
    real_ask = orch.ask

    def checked_ask(question):
        assert runtime.LOCK.locked()
        return real_ask(question)

    monkeypatch.setattr(orch, "ask", checked_ask)
    tools.ask_finsight("Should I buy AMD?")
    assert not runtime.LOCK.locked()
