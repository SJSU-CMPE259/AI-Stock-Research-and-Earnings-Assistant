"""Data agent: load one company's REAL data into shared memory.

Numbers route: quarterly revenue, gross_profit and operating_income from the XBRL store
(finsight.datastore, SEC companyfacts). If the ticker is not in the store yet it is
fetched with the existing EDGAR/XBRL loader and saved, then read back. Every value keeps
its ticker, fiscal period, accession, XBRL concept, filed date and data_mode="xbrl".

Sentiment/text route: passages from the asked ticker's latest 10-K via the existing EDGAR
fetch (backend.app.tools), filtered to that company's latest filing and sections, with
data_mode="filing". (The RAG/Qdrant retriever is used read-only when available; it is not
required here and is never built from this agent.)

No sample data is ever presented as a company's data. The bundled AMD sample is used ONLY
in offline/unit-test mode (FakeLLM) and is always stamped data_mode="sample" so the report
text can say so. If real data is missing for a ticker, the item is omitted and the answer
text must state it — never a silent substitution.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from agent_core.base import Agent
from agent_core.memory import SharedMemory

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "sample_amd.json"
SAMPLE_TICKER = "AMD"
METRICS = ("revenue", "gross_profit", "operating_income")


def load_company_data(path: Path = DATA_FILE) -> dict:
    """The bundled AMD sample dict (offline/test fallback only)."""
    return json.loads(Path(path).read_text())


def _finsight_src_on_path() -> bool:
    """Put the repo's src/ (finsight data layer) on sys.path, wherever it sits."""
    for base in Path(__file__).resolve().parents:
        cand = base / "src"
        if (cand / "finsight" / "datastore" / "api.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return True
    return False


def _store_rows(ticker: str, metric: str, n: int, conn) -> list[dict]:
    from finsight.datastore import api
    return api.get_metric(ticker, metric, "quarter", n, conn=conn)


@lru_cache(maxsize=8)
def fetch_financials(ticker: str, n: int = 6) -> dict:
    """Real quarterly revenue, gross_profit, operating_income from the XBRL store.

    Ingests the ticker (facts only) on a store miss, then reads back. Returns
    {metric: [rows...]} oldest→newest with QoQ/YoY and provenance. Raises if revenue
    is unavailable even after ingest. Network call — not used in offline/test mode.
    """
    if not _finsight_src_on_path():
        raise RuntimeError("finsight data layer (src/finsight) not found")
    from finsight.datastore import db, ingest

    conn = db.connect()
    if not _store_rows(ticker, "revenue", 1, conn):
        ingest.ingest_company(conn, ticker, download_docs=False)  # fetch SEC companyfacts, save
    out: dict[str, list[dict]] = {}
    for metric in METRICS:
        rows = _store_rows(ticker, metric, n, conn)
        out[metric] = [_num_item(ticker, metric, r) for r in reversed(rows)]  # oldest→newest
    if not out["revenue"]:
        raise RuntimeError(f"no XBRL revenue in store for {ticker}")
    return out


def _num_item(ticker: str, metric: str, r: dict) -> dict:
    """One financial fact with full provenance, in the shape the analysis/viz agents use."""
    return {
        "ticker": ticker.upper(), "metric": metric,
        "fiscal_year": r["fiscal_year"], "fiscal_period": r["fiscal_period"],
        "fiscal_label": r["fiscal_label"], "period_end": r["period_end"],
        "value_usd": float(r["value"]),
        "change_qoq_pct": r.get("change_vs_prior_quarter_pct"),
        "change_yoy_pct": r.get("change_vs_prior_year_pct"),
        "derived": bool(r.get("derived")),
        "data_mode": "xbrl",
        "source": {
            "form": r.get("form"), "accession": r.get("accession"),
            "concept": r.get("concept"), "filed": r.get("filed"), "url": r.get("source_url"),
        },
    }


@lru_cache(maxsize=16)
def load_filing_passages(ticker: str, form: str = "10-K", max_passages: int = 8) -> tuple:
    """Up to `max_passages` passages from the asked ticker's latest 10-K (Risk Factors +
    MD&A), each stamped with ticker, form, accession, section, filed and data_mode="filing".
    Cached per ticker. Raises on failure."""
    from backend.app.tools.edgar import get_edgar
    from backend.app.tools.sections import parse_filing

    edgar = get_edgar()
    filings = edgar.list_filings(ticker, form=form, limit=1)
    if not filings:
        raise RuntimeError(f"No {form} found for {ticker} on SEC EDGAR")
    f = filings[0]
    sections = parse_filing(edgar.download_filing(f), f.form)
    filed = getattr(f, "filing_date", None) or getattr(f, "filed", None)
    wanted = {"Item 1A": "Risk Factors", "Item 7": "MD&A"}
    per = max(1, max_passages // 2)
    out: list[dict] = []
    for s in sections:
        label = next((v for k, v in wanted.items() if s.name.startswith(k)), None)
        if not label:
            continue
        taken = 0
        for sent in re.split(r"(?<=[.!?])\s+", s.text):
            sent = sent.strip().replace("\n", " ")
            if 80 <= len(sent) <= 400:
                out.append({
                    "text": sent, "ticker": ticker.upper(), "section": label,
                    "data_mode": "filing",
                    "source": {"form": f.form, "accession": f.accession, "section": label,
                               "filed": str(filed) if filed else None, "url": f.url},
                })
                taken += 1
                if taken >= per:
                    break
    if not out:
        raise RuntimeError(f"Could not extract passages from {ticker} {f.form}")
    return tuple(out[:max_passages])


class DataAgent(Agent):
    name = "data_agent"
    reads: list[str] = []
    writes = ["amd_revenue_quarterly", "amd_gross_profit_quarterly",
              "amd_operating_income_quarterly", "amd_passages"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        tk = (ticker or SAMPLE_TICKER).upper()
        offline = "Fake" in type(llm).__name__  # unit tests / offline: no network
        wrote: list[str] = []

        fin, fin_mode = self._financials(tk, offline)
        if fin and fin.get("revenue"):
            key_for = {"revenue": "amd_revenue_quarterly",
                       "gross_profit": "amd_gross_profit_quarterly",
                       "operating_income": "amd_operating_income_quarterly"}
            for metric, key in key_for.items():
                rows = fin.get(metric) or []
                if rows:
                    memory.put(key, rows, self.name,
                               sources=[r["source"] for r in rows], freshness="stored",
                               description=f"{len(rows)}q {metric} for {tk} ({fin_mode})")
            wrote.append(f"financials ({len(fin['revenue'])}q, {fin_mode})")

        passages, p_mode = self._passages(tk, offline)
        if passages:
            memory.put("amd_passages", passages, self.name,
                       sources=[p["source"] for p in passages], freshness="stored",
                       description=f"{len(passages)} passages for {tk} ({p_mode})")
            wrote.append(f"amd_passages ({len(passages)}, {p_mode})")

        if not wrote:
            return f"{self.name}: {tk} -> no real data available (nothing written)"
        return f"{self.name}: {tk} -> wrote {', '.join(wrote)}"

    @staticmethod
    def _financials(tk: str, offline: bool) -> tuple[dict | None, str]:
        """(financials, data_mode). Real XBRL store for any company; AMD sample only offline."""
        if not offline:
            try:
                return fetch_financials(tk), "xbrl"
            except Exception:
                return None, ""
        if tk == SAMPLE_TICKER:
            rows = [dict(e, ticker=SAMPLE_TICKER, metric="revenue", data_mode="sample")
                    for e in load_company_data()["revenue_quarterly"]]
            return {"revenue": rows, "gross_profit": [], "operating_income": []}, "sample"
        return None, ""

    @staticmethod
    def _passages(tk: str, offline: bool) -> tuple[list | None, str]:
        """(passages, data_mode). Real latest-10-K passages for any company; AMD sample only offline."""
        if not offline:
            try:
                return list(load_filing_passages(tk)), "filing"
            except Exception:
                return None, ""
        if tk == SAMPLE_TICKER:
            ps = [dict(p, ticker=SAMPLE_TICKER, data_mode="sample")
                  for p in load_company_data()["passages"]]
            return ps, "sample"
        return None, ""
