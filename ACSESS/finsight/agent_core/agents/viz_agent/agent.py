"""Viz agent: turn the quarterly revenue series into a chart spec for the UI.

Headless by design — it writes a plain chart spec to memory under 'amd_chart'; the
portal (or any client) renders it. No plotting library imported here.
"""
from __future__ import annotations

from agent_core.base import Agent
from agent_core.memory import SharedMemory

_Q = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


class VizAgent(Agent):
    name = "viz_agent"
    reads = ["amd_revenue_quarterly"]
    writes = ["amd_chart"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        self.check_inputs(memory)
        rows = sorted(
            memory.get("amd_revenue_quarterly"),
            key=lambda e: (e["fiscal_year"], _Q[e["fiscal_period"]]),
        )
        points = [
            {"period": r["fiscal_label"], "revenue_b": round(r["value_usd"] / 1e9, 2)}
            for r in rows
        ]
        note = ""
        if memory.has("amd_notes"):
            notes = memory.get("amd_notes")
            if notes:
                note = notes[0]["text"]

        memory.put(
            "amd_chart",
            {
                "type": "bar",
                "title": f"{ticker or 'AMD'} quarterly revenue (sample data)",
                "x_label": "fiscal period",
                "y_label": "revenue ($B)",
                "points": points,
                "note": note,
            },
            self.name,
            description=f"revenue chart ({len(points)} quarters)",
        )
        return f"{self.name}: read amd_revenue_quarterly -> wrote amd_chart ({len(points)} bars)"
