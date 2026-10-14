"""Market agent: real-time quote. Reads the sample quote now; a live API later.

Writes amd_quote with freshness="real_time" so the orchestrator expires it after 60s
and never treats it as a stored batch fact.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agent_core.agents.data_agent.agent import load_company_data
from agent_core.base import Agent
from agent_core.memory import SharedMemory


class MarketAgent(Agent):
    name = "market_agent"
    reads: list[str] = []
    writes = ["amd_quote"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        quote = load_company_data()["quote"]  # sample value; swap for a live API later
        currency = quote.get("currency", "USD")
        memory.put(
            "amd_quote",
            {
                "price": quote["price"],
                "currency": currency,
                "as_of": datetime.now(timezone.utc).isoformat(),
                "real_time": True,
            },
            self.name,
            freshness="real_time",
            description=f"quote {quote['price']} {currency} (real_time)",
        )
        return f"{self.name}: wrote amd_quote ({quote['price']} {currency}, real_time)"
