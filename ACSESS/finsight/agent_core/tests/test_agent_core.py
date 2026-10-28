"""Agent-core tests. Offline only: FakeLLM, no network, no GPU."""
import json

from agent_core.agents.report_agent.agent import ReportAgent
from agent_core.llm import FakeLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator, route


# ---- routing (incl. the buy/sell refusal) --------------------------------

def test_route_refusal():
    assert route("Should I buy AMD?") == "refuse"
    assert route("should i invest in AMD right now") == "refuse"


def test_route_paths():
    assert route("What is AMD's stock price right now?") == "real_time"
    assert route("What was AMD's revenue last quarter?") == "numbers"
    assert route("What is the tone of AMD's filings?") == "sentiment"
    assert route("Tell me about AMD's business") == "text"


def test_refusal_runs_no_agents_and_no_llm():
    llm = FakeLLM()
    orch = Orchestrator(llm, memory=SharedMemory(), conversation=Conversation())
    tr = orch.ask("Should I buy AMD?")
    assert tr.route == "refuse"
    assert "investment advice" in tr.answer.lower()
    assert llm.calls == []  # the LLM was never called


# ---- memory put/get + save -----------------------------------------------

def test_memory_put_get_list():
    m = SharedMemory()
    m.put("amd_notes", {"a": 1}, "analysis_agent", description="d")
    assert m.get("amd_notes") == {"a": 1}
    assert m.has("amd_notes")
    assert "amd_notes" in m.list()
    assert "amd_notes" in m.describe()


def test_memory_save(tmp_path):
    m = SharedMemory()
    m.put("amd_quote", {"price": 1.0}, "market_agent", freshness="real_time")
    path = tmp_path / "mem.json"
    m.save(path)
    data = json.loads(path.read_text())
    assert data["amd_quote"]["value"] == {"price": 1.0}
    assert data["amd_quote"]["written_by"] == "market_agent"
    assert data["amd_quote"]["freshness"] == "real_time"


# ---- conversation: ticker remembered + stored reuse on Q2 -----------------

def test_ticker_remembered_and_stored_data_reused_on_q2():
    orch = Orchestrator(FakeLLM(), memory=SharedMemory(), conversation=Conversation())
    t1 = orch.ask("What was AMD's revenue last quarter?")
    assert t1.ticker == "AMD"

    t2 = orch.ask("How did it change from the year before?")  # no ticker named
    assert t2.ticker == "AMD"  # remembered from Q1
    assert "amd_revenue_quarterly" in t2.reused  # data_agent skipped, data reused


# ---- real_time quote expires after 60s ------------------------------------

def test_quote_reused_while_fresh_then_expires():
    mem = SharedMemory()
    orch = Orchestrator(FakeLLM(), memory=mem, conversation=Conversation())

    orch.ask("What is AMD's stock price right now?")
    assert mem.has("amd_quote")

    fresh = orch.ask("AMD price today?")
    assert "amd_quote" in fresh.reused  # < 60s -> reused

    mem.item("amd_quote").created_at -= 61  # age it past the TTL
    expired = orch.ask("AMD current price?")
    assert "amd_quote" not in expired.reused  # re-fetched


# ---- report check: fake IDs and fabricated numbers ------------------------

def test_report_check_catches_fake_id():
    ok, reason = ReportAgent._check("Revenue 7.10 [N9].", {"N1"}, "Revenue 7.10")
    assert not ok and "N9" in reason


def test_report_check_catches_fabricated_number():
    ok, reason = ReportAgent._check("Revenue 99.9 [N1].", {"N1"}, "Revenue 7.10")
    assert not ok and "99.9" in reason


def test_report_check_accepts_valid_combined_citation():
    ok, _ = ReportAgent._check("Revenue 7.10 [N1, N2].", {"N1", "N2"}, "Revenue 7.10 grew")
    assert ok


def test_report_falls_back_to_template_when_llm_fabricates():
    mem = SharedMemory()
    mem.put(
        "amd_notes",
        [{"id": "N1", "text": "Revenue FY2025 Q2: $7.10B.", "values": {"value_usd": 7100000000}, "source": {}}],
        "analysis_agent",
    )
    # Both attempts invent numbers that aren't in the notes -> template fallback.
    bad_llm = FakeLLM(["Revenue was $99.99B [N1].", "Revenue was $88.88B [N1]."])
    ReportAgent().run("AMD", "revenue?", mem, bad_llm)
    answer = mem.get("answer")
    assert "(sample data)" in answer
    assert "99.99" not in answer and "88.88" not in answer  # fabricated values rejected
    assert "7.10" in answer  # template uses the real note


def test_on_step_callback_and_events_backward_compatible():
    """ask(on_step=...) fires per step and fills trace.events; default call unchanged."""
    seen = []
    orch = Orchestrator(FakeLLM(), memory=SharedMemory(), conversation=Conversation())
    tr = orch.ask("What was AMD's revenue last quarter?", on_step=lambda e: seen.append(e["agent"]))
    # router fires first, then the numbers-route agents
    assert seen and seen[0] == "router"
    assert "report_agent" in seen
    # events mirror the callback and carry status + timing
    assert [e["agent"] for e in tr.events] == seen
    assert all("status" in e and "ms" in e for e in tr.events)
    # a callback that raises must never break the run
    tr2 = orch.ask("What is AMD's revenue?", on_step=lambda e: (_ for _ in ()).throw(ValueError("boom")))
    assert tr2.answer
    # default (no callback) still works and still records events
    tr3 = Orchestrator(FakeLLM()).ask("What is the tone of AMD's filings?")
    assert tr3.events and tr3.answer
