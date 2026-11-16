"""Data agent: load company data into shared memory.

Revenue (numbers route) is real SEC XBRL company-facts for any company, via Anna's data
layer (finsight.tools.xbrl); the bundled AMD sample is the offline fallback. Passages
(sentiment/text routes) come from the latest EDGAR filing, or the AMD sample.
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


def load_company_data(path: Path = DATA_FILE) -> dict:
    """Return the bundled AMD sample data dict."""
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=16)
def load_filing_passages(ticker: str, form: str = "10-K", max_passages: int = 8) -> tuple:
    """Fetch a company's latest filing and pull sentiment-relevant passages from
    Risk Factors (Item 1A) and MD&A (Item 7). Cached per ticker. Raises on failure."""
    from backend.app.tools.edgar import get_edgar
    from backend.app.tools.sections import parse_filing

    edgar = get_edgar()
    filings = edgar.list_filings(ticker, form=form, limit=1)
    if not filings:
        raise RuntimeError(f"No {form} found for {ticker} on SEC EDGAR")
    f = filings[0]
    sections = parse_filing(edgar.download_filing(f), f.form)
    src = {"form": f.form, "accession": f.accession, "url": f.url}
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
                out.append({"text": sent, "section": label, "source": src})
                taken += 1
                if taken >= per:
                    break
    if not out:
        raise RuntimeError(f"Could not extract passages from {ticker} {f.form}")
    return tuple(out[:max_passages])


def _finsight_src_on_path() -> bool:
    """Put the repo's src/ (Anna's finsight data layer) on sys.path, wherever it sits."""
    for base in Path(__file__).resolve().parents:
        cand = base / "src"
        if (cand / "finsight" / "tools" / "xbrl.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return True
    return False


@lru_cache(maxsize=16)
def fetch_revenue_quarterly(ticker: str, limit: int = 6) -> tuple:
    """Real quarterly revenue from SEC XBRL (finsight.tools.xbrl), oldest→newest, last `limit`.

    Returns the shape the analysis/viz agents expect; raises on any failure so the caller
    can fall back. Cached per ticker. Live network call — not used in offline/test mode.
    """
    if not _finsight_src_on_path():
        raise RuntimeError("finsight data layer (src/finsight) not found")
    from finsight.tools.xbrl import get_metric

    facts = get_metric(ticker, "revenue", "quarterly", limit=limit)
    rows = sorted(facts, key=lambda f: f.end)  # oldest -> newest
    if not rows:
        raise RuntimeError(f"no XBRL revenue facts for {ticker}")
    return tuple(
        {
            "fiscal_year": f.fiscal_year,
            "fiscal_period": f.fiscal_period,
            "fiscal_label": f"FY{f.fiscal_year} {f.fiscal_period}",
            "period_end": f.end,
            "value_usd": float(f.value),
            "source": {"form": f.form, "accession": f.accession, "url": f.source_url},
        }
        for f in rows
    )


class DataAgent(Agent):
    name = "data_agent"
    reads: list[str] = []
    writes = ["revenue_quarterly", "passages"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        tk = (ticker or SAMPLE_TICKER).upper()
        offline = "Fake" in type(llm).__name__  # unit tests / offline mode: no network
        wrote: list[str] = []

        revenue, rev_src = self._revenue(tk, offline)
        if revenue:
            memory.put(
                "revenue_quarterly", list(revenue), self.name,
                sources=[e["source"] for e in revenue], freshness="stored",
                description=f"{len(revenue)} quarters of revenue for {tk} ({rev_src})",
            )
            wrote.append(f"revenue_quarterly ({len(revenue)}q, {rev_src})")

        if tk == SAMPLE_TICKER and (offline or revenue is None or rev_src == "sample"):
            passages = load_company_data()["passages"]
        else:
            passages = list(load_filing_passages(tk))
        memory.put(
            "passages", passages, self.name,
            sources=[p["source"] for p in passages], freshness="stored",
            description=f"{len(passages)} filing passages for {tk}",
        )
        wrote.append(f"passages ({len(passages)})")
        return f"{self.name}: {tk} -> wrote {', '.join(wrote)}"

    @staticmethod
    def _revenue(tk: str, offline: bool) -> tuple:
        """(rows, source) — real XBRL for any company, AMD sample as the offline/failure fallback."""
        if not offline:
            try:
                return fetch_revenue_quarterly(tk), "SEC XBRL"
            except Exception:
                pass
        if tk == SAMPLE_TICKER:
            return tuple(load_company_data()["revenue_quarterly"]), "sample"
        return None, ""
