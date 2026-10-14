"""Analysis agent: compute QoQ / YoY / trend notes in Python (never the LLM).

Writes amd_notes as a list of {id, text, values, source}. The report agent later
cites these by id (e.g. [N1]) and must copy the numbers exactly.
"""
from __future__ import annotations

from agent_core.base import Agent
from agent_core.memory import SharedMemory

_Q_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _fmt_usd(v: float) -> str:
    return f"${v / 1e9:.2f}B"


def _pct(new: float, old: float) -> float:
    return round((new - old) / old * 100, 1)


class AnalysisAgent(Agent):
    name = "analysis_agent"
    reads = ["amd_revenue_quarterly"]
    writes = ["amd_notes"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        self.check_inputs(memory)
        rows = sorted(
            memory.get("amd_revenue_quarterly"),
            key=lambda e: (e["fiscal_year"], _Q_ORDER[e["fiscal_period"]]),
        )
        latest = rows[-1]
        notes: list[dict] = []

        if len(rows) >= 2:
            prev = rows[-2]
            qoq = _pct(latest["value_usd"], prev["value_usd"])
            notes.append({
                "id": f"N{len(notes) + 1}",
                "text": (
                    f"Revenue {latest['fiscal_label']}: {_fmt_usd(latest['value_usd'])}, "
                    f"{'up' if qoq >= 0 else 'down'} {abs(qoq)}% QoQ from "
                    f"{prev['fiscal_label']} ({_fmt_usd(prev['value_usd'])})."
                ),
                "values": {
                    "fiscal_label": latest["fiscal_label"],
                    "value_usd": latest["value_usd"],
                    "prev_label": prev["fiscal_label"],
                    "prev_value_usd": prev["value_usd"],
                    "qoq_pct": qoq,
                },
                "source": latest["source"],
            })

        yoy_prev = next(
            (e for e in rows
             if e["fiscal_period"] == latest["fiscal_period"]
             and e["fiscal_year"] == latest["fiscal_year"] - 1),
            None,
        )
        if yoy_prev:
            yoy = _pct(latest["value_usd"], yoy_prev["value_usd"])
            notes.append({
                "id": f"N{len(notes) + 1}",
                "text": (
                    f"Revenue {latest['fiscal_label']}: {_fmt_usd(latest['value_usd'])}, "
                    f"{'up' if yoy >= 0 else 'down'} {abs(yoy)}% YoY from "
                    f"{yoy_prev['fiscal_label']} ({_fmt_usd(yoy_prev['value_usd'])})."
                ),
                "values": {
                    "fiscal_label": latest["fiscal_label"],
                    "value_usd": latest["value_usd"],
                    "prior_year_label": yoy_prev["fiscal_label"],
                    "prior_year_value_usd": yoy_prev["value_usd"],
                    "yoy_pct": yoy,
                },
                "source": latest["source"],
            })

        first = rows[0]
        trend = _pct(latest["value_usd"], first["value_usd"])
        notes.append({
            "id": f"N{len(notes) + 1}",
            "text": (
                f"Revenue rose from {_fmt_usd(first['value_usd'])} ({first['fiscal_label']}) "
                f"to {_fmt_usd(latest['value_usd'])} ({latest['fiscal_label']}), "
                f"{'+' if trend >= 0 else ''}{trend}% over {len(rows)} quarters."
            ),
            "values": {
                "first_label": first["fiscal_label"],
                "first_value_usd": first["value_usd"],
                "latest_label": latest["fiscal_label"],
                "latest_value_usd": latest["value_usd"],
                "change_pct": trend,
                "n_quarters": len(rows),
            },
            "source": latest["source"],
        })

        memory.put(
            "amd_notes", notes, self.name,
            sources=[n["source"] for n in notes],
            description=f"{len(notes)} computed notes (QoQ, YoY, trend)",
        )
        headline = notes[0]["text"] if notes else "no notes"
        return f"{self.name}: read amd_revenue_quarterly -> wrote amd_notes ({len(notes)} notes); {headline}"
