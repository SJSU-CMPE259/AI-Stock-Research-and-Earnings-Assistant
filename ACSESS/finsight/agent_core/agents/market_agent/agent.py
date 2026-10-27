"""Market agent: real-time quote. Uses Anna's live Finnhub price tool when a
FINNHUB_API_KEY is configured, and falls back to the sample quote otherwise.

Writes amd_quote with freshness="real_time" so the orchestrator expires it after 60s
and never treats it as a stored batch fact.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agent_core.agents.data_agent.agent import load_company_data
from agent_core.base import Agent
from agent_core.memory import SharedMemory
from agent_core.tools import news, prices


class MarketAgent(Agent):
    name = "market_agent"
    reads: list[str] = []
    writes = ["amd_quote", "amd_news"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        sym = (ticker or "AMD").upper()
        quote_msg = self._quote(sym, memory)
        news_msg = self._news(sym, memory)
        return f"{self.name}: {quote_msg}{news_msg}"

    def _quote(self, sym: str, memory: SharedMemory) -> str:
        # Try the live Finnhub quote; fall back to sample data on any failure
        # (no FINNHUB_API_KEY, network error, unknown ticker, rate limit).
        try:
            q = prices.get_quote(sym)
            memory.put(
                "amd_quote",
                {
                    "price": q.price,
                    "currency": "USD",
                    "as_of": q.as_of,
                    "change_pct": q.change_pct,
                    "source": q.source,
                    "source_url": q.source_url,
                    "summary": q.summary(),
                    "market_open_likely": q.market_open_likely,
                    "live": True,
                    "real_time": True,
                },
                self.name,
                freshness="real_time",
                description=f"live quote {q.summary()}",
            )
            return f"wrote amd_quote (live — {q.summary()})"
        except Exception as e:  # noqa: BLE001 — degrade to sample, never break the route
            quote = load_company_data()["quote"]
            currency = quote.get("currency", "USD")
            memory.put(
                "amd_quote",
                {
                    "price": quote["price"],
                    "currency": currency,
                    "as_of": quote.get("as_of", datetime.now(timezone.utc).isoformat()),
                    "live": False,
                    "real_time": True,
                },
                self.name,
                freshness="real_time",
                description=f"sample quote {quote['price']} {currency} (live price unavailable)",
            )
            reason = type(e).__name__
            return (f"wrote amd_quote (sample — {quote['price']} {currency}; "
                    f"live price unavailable: {reason})")

    def _news(self, sym: str, memory: SharedMemory) -> str:
        """Attach up to 3 recent headlines (live Finnhub) for the report to cite.

        Best-effort: no key / network error / no headlines -> write nothing, so the
        real_time route still answers with the price alone.
        """
        try:
            articles = news.get_company_news(sym, days=7, limit=3, min_relevance="medium")
        except Exception as e:  # noqa: BLE001 — news is optional enrichment
            return f"; no news ({type(e).__name__})"
        if not articles:
            return "; no recent news"
        memory.put(
            "amd_news",
            [{"headline": a.headline, "source": a.source, "date": a.date, "url": a.url}
             for a in articles],
            self.name,
            freshness="real_time",
            description=f"{len(articles)} recent headlines",
        )
        return f"; wrote amd_news ({len(articles)} headlines)"
