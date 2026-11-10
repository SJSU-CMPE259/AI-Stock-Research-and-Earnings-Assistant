"""Rule-based orchestrator: route a question, run agents in order, return a trace.

Routing uses simple Python keyword rules (Mistral-7B is weak at tool calling), applied
in priority order. The LLM is never used to route.
"""
from __future__ import annotations

import re
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
    "filings."
)

# whole-word advice match so "share buybacks", "largest shareholders" and
# "selling, general and administrative" are NOT mistaken for buy/sell/hold advice.
_ADVICE = re.compile(r"\b(buy|sell|hold|invest|recommend)\b", re.IGNORECASE)

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
    # structured, UI-friendly trace: one event per router/agent step
    events: list[dict] = field(default_factory=list)


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
    if _ADVICE.search(q):
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

    def ask(self, question: str, on_step=None) -> Trace:
        """Route the question and run its agents. `on_step(event)` (optional) is called
        once per router/agent step for live progress; it never affects the result."""
        start = time.time()
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        prev_ticker = self.conversation.current_ticker
        ticker = resolve_ticker(question, self.conversation)
        r, source, conf = self._route(question)
        trace = Trace(run_id=run_id, question=question, ticker=ticker, route=r,
                      route_source=source, route_confidence=conf)
        conf_txt = f", conf {conf:.2f}" if conf is not None else ""
        trace.steps.append(f"router [{source}]: {r}{conf_txt}")
        self._emit(trace, on_step, {"agent": "router", "status": "done", "ms": 0.0,
                                    "route": r, "source": source, "confidence": conf, "wrote": []})

        if r == "refuse":
            trace.answer = REFUSAL
            trace.steps.append("-> refused (no agents, no LLM)")
            return self._finish(trace, start, question)

        if ticker:
            self.conversation.current_ticker = ticker

        # A new ticker must not reuse or cite the previous ticker's facts: drop the
        # ticker-scoped memory (data/quote/news/sentiment/charts) when the ticker changes.
        if ticker and prev_ticker and ticker != prev_ticker:
            dropped = self.memory.forget_ticker_scope()
            if dropped:
                trace.steps.append(
                    f"ticker changed {prev_ticker} -> {ticker}: cleared {', '.join(dropped)}"
                )

        self.memory.put(
            "conversation_state", self.conversation.state_text(), "orchestrator",
            description="recent company/period + last Q/A pairs",
        )

        for agent_cls in ROUTES[r]:
            agent = agent_cls()

            if self._reuse_stored_data(agent, trace):
                self._emit(trace, on_step, {"agent": agent.name, "status": "reused",
                                            "ms": 0.0, "wrote": list(agent.writes)})
                continue
            if self._reuse_fresh_quote(agent, trace, ticker):
                self._emit(trace, on_step, {"agent": agent.name, "status": "reused",
                                            "ms": 0.0, "wrote": ["amd_quote"]})
                continue

            t0 = time.time()
            try:
                detail = agent.run(ticker, question, self.memory, self.llm)
                ms = (time.time() - t0) * 1000
                trace.steps.append(detail)
                self._emit(trace, on_step, {"agent": agent.name, "status": "done",
                                            "ms": round(ms, 1), "wrote": list(agent.writes),
                                            "detail": detail})
            except Exception as e:
                ms = (time.time() - t0) * 1000
                trace.steps.append(f"{agent.name}: FAILED — {e}")
                self._emit(trace, on_step, {"agent": agent.name, "status": "error",
                                            "ms": round(ms, 1), "wrote": [], "detail": str(e)})
                trace.answer = f"Stopped: {agent.name} failed ({e})."
                return self._finish(trace, start, question)

        self._remember_period()
        trace.answer = self.memory.get("answer") or "(no answer written)"
        return self._finish(trace, start, question)

    @staticmethod
    def _emit(trace: Trace, on_step, event: dict) -> None:
        """Record a structured event and notify the optional callback (never fatal)."""
        trace.events.append(event)
        if on_step is not None:
            try:
                on_step(event)
            except Exception:  # noqa: BLE001 — a UI callback must never break a run
                pass

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

    def _reuse_fresh_quote(self, agent, trace: Trace, ticker: str | None) -> bool:
        """Skip market_agent only if a quote for THIS ticker is younger than QUOTE_TTL_S.

        Must match on ticker: a fresh AMD quote must not be reused for an NVDA question.
        """
        if not isinstance(agent, MarketAgent) or not self.memory.has("amd_quote"):
            return False
        quote = self.memory.get("amd_quote")
        if not isinstance(quote, dict) or quote.get("ticker") != ticker:
            return False  # different (or unknown) ticker -> fetch fresh
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
