"""Analysis agent: compute QoQ / YoY / trend / margin notes in Python (never the LLM).

Writes amd_notes as a list of {id, text, values, source, ticker, data_mode}. The report
agent cites these by id (e.g. [N1]) and must copy the numbers exactly. Margins (gross,
operating) are computed from the XBRL gross_profit / operating_income series the data
agent loads; their change is reported in percentage points (pp), not percent.
"""
from __future__ import annotations

from agent_core.base import Agent
from agent_core.memory import SharedMemory

_Q_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _fmt_usd(v: float) -> str:
    return f"${v / 1e9:.2f}B"


def _pct(new: float, old: float) -> float:
    return round((new - old) / old * 100, 1)


def _key(e: dict) -> tuple:
    return (e["fiscal_year"], _Q_ORDER[e["fiscal_period"]])


class AnalysisAgent(Agent):
    name = "analysis_agent"
    reads = ["amd_revenue_quarterly"]
    writes = ["amd_notes"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        self.check_inputs(memory)
        rows = sorted(memory.get("amd_revenue_quarterly"), key=_key)
        latest, first = rows[-1], rows[0]
        tk = latest.get("ticker", ticker)
        mode = latest.get("data_mode", "xbrl")
        notes: list[dict] = []

        def add(text: str, values: dict, source: dict) -> None:
            notes.append({"id": f"N{len(notes) + 1}", "text": text, "values": values,
                          "source": source, "ticker": tk, "data_mode": mode})

        if len(rows) >= 2:
            prev = rows[-2]
            qoq = _pct(latest["value_usd"], prev["value_usd"])
            add(f"Revenue {latest['fiscal_label']}: {_fmt_usd(latest['value_usd'])}, "
                f"{'up' if qoq >= 0 else 'down'} {abs(qoq)}% QoQ from {prev['fiscal_label']} "
                f"({_fmt_usd(prev['value_usd'])}).",
                {"fiscal_label": latest["fiscal_label"], "value_usd": latest["value_usd"],
                 "prev_label": prev["fiscal_label"], "prev_value_usd": prev["value_usd"],
                 "qoq_pct": qoq}, latest["source"])

        yoy_prev = next((e for e in rows if e["fiscal_period"] == latest["fiscal_period"]
                         and e["fiscal_year"] == latest["fiscal_year"] - 1), None)
        if yoy_prev:
            yoy = _pct(latest["value_usd"], yoy_prev["value_usd"])
            add(f"Revenue {latest['fiscal_label']}: {_fmt_usd(latest['value_usd'])}, "
                f"{'up' if yoy >= 0 else 'down'} {abs(yoy)}% YoY from {yoy_prev['fiscal_label']} "
                f"({_fmt_usd(yoy_prev['value_usd'])}).",
                {"fiscal_label": latest["fiscal_label"], "value_usd": latest["value_usd"],
                 "prior_year_label": yoy_prev["fiscal_label"],
                 "prior_year_value_usd": yoy_prev["value_usd"], "yoy_pct": yoy}, latest["source"])

        # Margins: gross and operating, in % of revenue, with QoQ change in percentage points.
        rev_by = {_key(e): e for e in rows}
        self._margin_note(memory, "amd_gross_profit_quarterly", "Gross margin", rev_by, add)
        self._margin_note(memory, "amd_operating_income_quarterly", "Operating margin", rev_by, add)

        trend = _pct(latest["value_usd"], first["value_usd"])
        add(f"Revenue rose from {_fmt_usd(first['value_usd'])} ({first['fiscal_label']}) to "
            f"{_fmt_usd(latest['value_usd'])} ({latest['fiscal_label']}), "
            f"{'+' if trend >= 0 else ''}{trend}% over {len(rows)} quarters.",
            {"first_label": first["fiscal_label"], "first_value_usd": first["value_usd"],
             "latest_label": latest["fiscal_label"], "latest_value_usd": latest["value_usd"],
             "change_pct": trend, "n_quarters": len(rows)}, latest["source"])

        memory.put("amd_notes", notes, self.name, sources=[n["source"] for n in notes],
                   description=f"{len(notes)} computed notes (QoQ, YoY, margins, trend) for {tk}")
        headline = notes[0]["text"] if notes else "no notes"
        return f"{self.name}: {tk} -> wrote amd_notes ({len(notes)} notes); {headline}"

    @staticmethod
    def _margin_note(memory, key, label, rev_by, add) -> None:
        """Margin = metric / revenue; QoQ change reported in percentage points."""
        series = sorted((memory.get(key) or []), key=_key)
        pts = [(e, rev_by[_key(e)]) for e in series if _key(e) in rev_by and rev_by[_key(e)]["value_usd"]]
        if not pts:
            return
        e, rev = pts[-1]
        m = round(e["value_usd"] / rev["value_usd"] * 100, 1)
        values = {"fiscal_label": e["fiscal_label"], "margin_pct": m,
                  "numerator_usd": e["value_usd"], "revenue_usd": rev["value_usd"]}
        text = f"{label} {e['fiscal_label']}: {m}%"
        if len(pts) >= 2:
            pe, prev_rev = pts[-2]
            pm = round(pe["value_usd"] / prev_rev["value_usd"] * 100, 1)
            pp = round(m - pm, 1)
            values.update({"prev_label": pe["fiscal_label"], "prev_margin_pct": pm, "qoq_change_pp": pp})
            text += (f", {'up' if pp >= 0 else 'down'} {abs(pp)} pp QoQ from "
                     f"{pe['fiscal_label']} ({pm}%).")
        else:
            text += "."
        add(text, values, e["source"])
