"""Rule-based orchestrator: route a question, run agents in order, return a trace.

Routing uses simple Python keyword rules (Mistral-7B is weak at tool calling), applied
in priority order. The LLM is never used to route.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from agent_core.agents.analysis_agent.agent import AnalysisAgent
from agent_core.agents.classify_agent.agent import ClassifyAgent
from agent_core.agents.data_agent.agent import DataAgent
from agent_core.agents.market_agent.agent import MarketAgent
from agent_core.agents.report_agent.agent import ReportAgent
from agent_core.agents.viz_agent.agent import VizAgent
from agent_core.memory import Conversation, SharedMemory

REFUSAL = (
    "I can't give investment advice or buy/sell/hold recommendations. I can share the "
    "underlying facts instead — for example revenue, margins, or the tone of recent "
    "filings. (sample data)"
)

KNOWN_TICKERS = {"AMD", "NVDA", "AAPL", "MSFT", "TSLA", "INTC"}

QUOTE_TTL_S = 60  # a real_time quote older than this must be re-fetched

ROUTES = {
    "real_time": [MarketAgent, ReportAgent],
    "numbers": [DataAgent, AnalysisAgent, VizAgent, ReportAgent],
    "sentiment": [DataAgent, ClassifyAgent, VizAgent, ReportAgent],
    "text": [DataAgent, ReportAgent],
}


@dataclass
class Trace:
    run_id: str
    question: str
    ticker: str | None
    route: str
    steps: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    answer: str = ""
    elapsed_s: float = 0.0
    route_source: str = "keyword"
    route_confidence: float | None = None


def resolve_ticker(question: str, conversation: Conversation) -> str | None:
    """Use a ticker named in the question, else fall back to the conversation's."""
    for tok in question.replace("?", " ").replace("'", " ").split():
        up = tok.upper().strip(".,")
        if up in KNOWN_TICKERS:
            return up
    return conversation.current_ticker


def route(question: str) -> str:
    """Pick a path by keyword, in priority order."""
    q = question.lower()
    if any(w in q for w in ["buy", "sell", "hold", "should i invest", "should i buy"]):
        return "refuse"
    if any(w in q for w in ["price", "right now", "today", "current"]):
        return "real_time"
    if any(w in q for w in ["revenue", "income", "eps", "margin", "growth", "last quarter", "change"]):
        return "numbers"
    if any(w in q for w in ["sentiment", "tone", "positive", "negative", "outlook"]):
        return "sentiment"
    return "text"


class Orchestrator:
    """Routes one question to a path and runs its agents in order over shared memory."""

    def __init__(self, llm, memory: SharedMemory | None = None,
                 conversation: Conversation | None = None, router=None):
        self.llm = llm
        self.memory = memory or SharedMemory()
        self.conversation = conversation or Conversation()
        self.router = router  # optional model router (e.g. LayaRouter); else keyword rules

    def _route(self, question: str) -> tuple[str, str, float | None]:
        """Return (route, source, confidence). Uses the model router if set, else keywords."""
        if self.router is not None:
            try:
                r, conf = self.router.route(question)
                return r, self.router.source, conf
            except Exception:
                pass  # endpoint down / error -> keyword fallback
        return route(question), "keyword", None

    def ask(self, question: str) -> Trace:
        start = time.time()
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        ticker = resolve_ticker(question, self.conversation)
        r, source, conf = self._route(question)
        trace = Trace(run_id=run_id, question=question, ticker=ticker, route=r,
                      route_source=source, route_confidence=conf)
        conf_txt = f", conf {conf:.2f}" if conf is not None else ""
        trace.steps.append(f"router [{source}]: {r}{conf_txt}")

        if r == "refuse":
            trace.answer = REFUSAL
            trace.steps.append("-> refused (no agents, no LLM)")
            return self._finish(trace, start, question)

        if ticker:
            self.conversation.current_ticker = ticker

        self.memory.put(
            "conversation_state", self.conversation.state_text(), "orchestrator",
            description="recent company/period + last Q/A pairs",
        )

        for agent_cls in ROUTES[r]:
            agent = agent_cls()

            if self._reuse_stored_data(agent, trace):
                continue
            if self._reuse_fresh_quote(agent, trace):
                continue

            try:
                trace.steps.append(agent.run(ticker, question, self.memory, self.llm))
            except Exception as e:
                trace.steps.append(f"{agent.name}: FAILED — {e}")
                trace.answer = f"Stopped: {agent.name} failed ({e})."
                return self._finish(trace, start, question)

        self._remember_period()
        trace.answer = self.memory.get("answer") or "(no answer written)"
        return self._finish(trace, start, question)

    def _reuse_stored_data(self, agent, trace: Trace) -> bool:
        """Skip data_agent if its stored outputs are already in memory."""
        if not isinstance(agent, DataAgent):
            return False
        if all(
            self.memory.has(w) and self.memory.item(w).freshness == "stored"
            for w in agent.writes
        ):
            trace.steps.append(
                f"{agent.name}: reused stored data ({', '.join(agent.writes)}) — skipped"
            )
            trace.reused.extend(agent.writes)
            return True
        return False

    def _reuse_fresh_quote(self, agent, trace: Trace) -> bool:
        """Skip market_agent if the real_time quote is younger than QUOTE_TTL_S."""
        if not isinstance(agent, MarketAgent) or not self.memory.has("amd_quote"):
            return False
        age = self.memory.age_seconds("amd_quote")
        if age is not None and age < QUOTE_TTL_S:
            trace.steps.append(
                f"{agent.name}: reused fresh quote (age {age:.0f}s < {QUOTE_TTL_S}s) — skipped"
            )
            trace.reused.append("amd_quote")
            return True
        return False

    def _remember_period(self) -> None:
        if self.memory.has("amd_notes"):
            notes = self.memory.get("amd_notes")
            if notes:
                self.conversation.current_period = notes[0]["values"].get("fiscal_label")

    def _finish(self, trace: Trace, start: float, question: str) -> Trace:
        trace.elapsed_s = time.time() - start
        self.conversation.add(question, trace.answer)
        return trace
