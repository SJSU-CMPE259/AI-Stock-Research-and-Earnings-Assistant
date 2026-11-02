"""Viz agent: build chart specs for the UI from whatever is in shared memory.

Headless by design — it writes plain chart specs (no plotting library here); the portal
renders them. Produces a revenue chart (numbers route) and/or a sentiment breakdown
(sentiment route), depending on what earlier agents wrote.
"""
from __future__ import annotations

from agent_core.base import Agent
from agent_core.memory import SharedMemory

_Q = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


class VizAgent(Agent):
    name = "viz_agent"
    reads: list[str] = []  # opportunistic: charts whatever data is present
    writes = ["amd_chart", "amd_sentiment_chart"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        wrote = []

        if memory.has("amd_revenue_quarterly"):
            rows = sorted(
                memory.get("amd_revenue_quarterly"),
                key=lambda e: (e["fiscal_year"], _Q[e["fiscal_period"]]),
            )
            points = [
                {"period": r["fiscal_label"], "revenue_b": round(r["value_usd"] / 1e9, 2)}
                for r in rows
            ]
            note = ""
            if memory.has("amd_notes") and memory.get("amd_notes"):
                note = memory.get("amd_notes")[0]["text"]
            memory.put(
                "amd_chart",
                {"type": "bar", "title": f"{ticker or 'AMD'} quarterly revenue (sample data)",
                 "x_label": "fiscal period", "y_label": "revenue ($B)",
                 "points": points, "note": note},
                self.name, description=f"revenue chart ({len(points)} quarters)",
            )
            wrote.append("amd_chart")

        if memory.has("amd_sentiment"):
            s = memory.get("amd_sentiment")
            counts = s.get("counts", {}) if isinstance(s, dict) else {}
            order = ["positive", "neutral", "negative", "uncertain"]
            points = [{"label": k, "count": int(counts.get(k, 0))}
                      for k in order if counts.get(k)]
            memory.put(
                "amd_sentiment_chart",
                {"type": "bar", "title": f"{ticker or 'AMD'} filing sentiment (sample data)",
                 "x_label": "sentiment", "y_label": "passages",
                 "points": points, "note": s.get("summary", "") if isinstance(s, dict) else ""},
                self.name, description=f"sentiment breakdown ({len(points)} labels)",
            )
            wrote.append("amd_sentiment_chart")

        return f"{self.name}: wrote {', '.join(wrote)}" if wrote else f"{self.name}: no chartable data"
