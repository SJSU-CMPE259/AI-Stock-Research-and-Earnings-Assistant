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


def test_fresh_quote_not_reused_across_tickers():
    """A fresh AMD quote must NOT be reused for an NVDA question (ticker must match)."""
    mem = SharedMemory()
    orch = Orchestrator(FakeLLM(), memory=mem, conversation=Conversation())

    orch.ask("What is AMD's stock price right now?")
    assert mem.get("amd_quote")["ticker"] == "AMD"

    nvda = orch.ask("What is NVDA's stock price right now?")  # within TTL, different ticker
    assert "amd_quote" not in nvda.reused  # re-fetched, not reused
    assert mem.get("amd_quote")["ticker"] == "NVDA"


# ---- report check: fake IDs and fabricated numbers ------------------------

def _note(fid, text, ticker="AMD"):
    return {fid: {"id": fid, "text": text, "search_text": text, "ticker": ticker, "kind": "note"}}


def test_report_check_catches_fake_id():
    ok, reason = ReportAgent._check("Revenue 7.10 [N9].", _note("N1", "Revenue 7.10"), "AMD", False)
    assert not ok and "N9" in reason


def test_report_check_catches_fabricated_number():
    ok, reason = ReportAgent._check("Revenue 99.9 [N1].", _note("N1", "Revenue 7.10"), "AMD", False)
    assert not ok and "99.9" in reason


def test_report_check_accepts_valid_combined_citation():
    by_id = {**_note("N1", "Revenue 7.10 grew"), **_note("N2", "Revenue 7.10 grew")}
    ok, _ = ReportAgent._check("Revenue 7.10 [N1, N2].", by_id, "AMD", False)
    assert ok


def test_report_check_rejects_cross_ticker():
    # entity safety (Task 3): a note from NVDA cannot support an AMD answer
    ok, reason = ReportAgent._check("Revenue 7.10 [N1].", _note("N1", "Revenue 7.10", "NVDA"), "AMD", False)
    assert not ok and "NVDA" in reason


def test_report_check_requires_tone_counts():
    # coverage (Task 4): a tone question must state the counts and cite [S1]
    by_id = {"S1": {"id": "S1", "search_text": "5 positive", "ticker": "AMD", "kind": "sentiment"}}
    assert not ReportAgent._check("The tone is positive.", by_id, "AMD", True)[0]
    assert ReportAgent._check("5 positive [S1].", by_id, "AMD", True)[0]


def test_report_falls_back_to_template_when_llm_fabricates():
    mem = SharedMemory()
    mem.put(
        "amd_notes",
        [{"id": "N1", "text": "Revenue FY2025 Q2: $7.10B.", "values": {"value_usd": 7100000000},
          "source": {}, "ticker": "AMD", "data_mode": "xbrl"}],
        "analysis_agent",
    )
    # Both attempts invent numbers that aren't in the notes -> template fallback.
    bad_llm = FakeLLM(["Revenue was $99.99B [N1].", "Revenue was $88.88B [N1]."])
    ReportAgent().run("AMD", "revenue?", mem, bad_llm)
    answer = mem.get("answer")
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


def test_router_no_false_refusals():
    """Whole-word advice match: 'buybacks', 'shareholders', 'selling, general...' aren't refused."""
    for q in ["Tell me about AMD's share buybacks",
              "Who are AMD's largest shareholders?",
              "What is selling, general and administrative expense?"]:
        assert route(q) != "refuse", q
    # genuine advice is still refused
    assert route("Should I buy AMD?") == "refuse"
    assert route("Should I sell NVDA?") == "refuse"
    assert route("Is AMD a hold?") == "refuse"


def test_refusal_text_has_no_sample_tag():
    from agent_core.orchestrator import REFUSAL
    assert "(sample data)" not in REFUSAL  # a refusal fetches no data


def test_ticker_change_clears_stale_scope():
    """A new ticker must not reuse the previous ticker's stored filing data."""
    orch = Orchestrator(FakeLLM(), memory=SharedMemory(), conversation=Conversation())
    orch.ask("What was AMD's revenue last quarter?")
    assert orch.memory.has("amd_revenue_quarterly")
    tr = orch.ask("What was NVDA's revenue last quarter?")  # switch ticker
    assert "amd_revenue_quarterly" not in tr.reused  # cleared, re-fetched (not reused from AMD)


def test_report_label_reflects_fake_engine():
    """In offline mode the step says 'fake', not 'mistral'."""
    orch = Orchestrator(FakeLLM(), memory=SharedMemory(), conversation=Conversation())
    tr = orch.ask("What was AMD's revenue last quarter?")
    report_step = next(s for s in tr.steps if s.startswith("report_agent"))
    assert "fake" in report_step and "mistral" not in report_step


# ---- real-data fixes: one passage numbering + no sample for other tickers --

def test_evidence_single_numbering():
    """[Pk] in the report prompt and in the Sources list come from ONE ordered list."""
    from agent_core.agents.report_agent.agent import build_evidence
    mem = SharedMemory()
    passages = [{"text": f"passage {i}", "ticker": "AMD", "source": {}, "data_mode": "filing"}
                for i in range(1, 6)]
    mem.put("amd_passages", passages, "data_agent")
    ev = build_evidence(mem, "q")
    ids = [e["id"] for e in ev if e["kind"] == "passage"]
    pmap = {e["id"]: e["text"] for e in ev if e["kind"] == "passage"}
    assert ids == ["P1", "P2", "P3", "P4", "P5"]  # stored order preserved
    assert pmap["P2"] == "passage 2" and pmap["P5"] == "passage 5"


def test_data_agent_offline_is_amd_sample_only():
    """Offline (FakeLLM): AMD -> sample stamped data_mode='sample'; other tickers -> nothing
    (never sample under another company's name — the K2 bug)."""
    from agent_core.agents.data_agent.agent import DataAgent
    nvda = SharedMemory()
    DataAgent().run("NVDA", "revenue?", nvda, FakeLLM())
    assert not nvda.has("amd_revenue_quarterly") and not nvda.has("amd_passages")
    amd = SharedMemory()
    DataAgent().run("AMD", "revenue?", amd, FakeLLM())
    rev = amd.get("amd_revenue_quarterly")
    assert rev and all(r["data_mode"] == "sample" and r["ticker"] == "AMD" for r in rev)
