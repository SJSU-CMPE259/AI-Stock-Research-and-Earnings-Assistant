"""Data agent: load company data into shared memory.

For now it loads the bundled sample file. A Data teammate later replaces only
`load_company_data()` with real SEC tools; the rest of the core is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_core.base import Agent
from agent_core.memory import SharedMemory

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "sample_amd.json"


def load_company_data(path: Path = DATA_FILE) -> dict:
    """Return the company data dict. Swap this for real SEC loading later."""
    return json.loads(Path(path).read_text())


class DataAgent(Agent):
    name = "data_agent"
    reads: list[str] = []
    writes = ["amd_revenue_quarterly", "amd_passages"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        data = load_company_data()
        revenue = data["revenue_quarterly"]
        passages = data["passages"]

        memory.put(
            "amd_revenue_quarterly", revenue, self.name,
            sources=[e["source"] for e in revenue],
            freshness="stored",
            description=f"{len(revenue)} quarters of revenue (fiscal labels)",
        )
        memory.put(
            "amd_passages", passages, self.name,
            sources=[p["source"] for p in passages],
            freshness="stored",
            description=f"{len(passages)} filing passages",
        )
        return (
            f"{self.name}: loaded sample_amd.json -> wrote "
            f"amd_revenue_quarterly ({len(revenue)} quarters), amd_passages ({len(passages)})"
        )
