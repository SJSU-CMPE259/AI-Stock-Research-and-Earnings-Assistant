"""Data agent: load company data into shared memory.

AMD uses the bundled sample file (fast, offline). For any other company we fetch its
latest SEC filing from EDGAR and extract passages, so the sentiment/text routes work
for real companies. Structured revenue (the numbers route) stays AMD-only for now —
that needs XBRL financial facts, which aren't wired in yet.
"""
from __future__ import annotations

import json
import re
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


class DataAgent(Agent):
    name = "data_agent"
    reads: list[str] = []
    writes = ["amd_revenue_quarterly", "amd_passages"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        tk = (ticker or SAMPLE_TICKER).upper()

        if tk != SAMPLE_TICKER:
            passages = list(load_filing_passages(tk))
            memory.put(
                "amd_passages", passages, self.name,
                sources=[p["source"] for p in passages], freshness="stored",
                description=f"{len(passages)} real filing passages from {tk} "
                            f"{passages[0]['source']['form']}",
            )
            return f"{self.name}: fetched {tk} filing from SEC EDGAR -> wrote amd_passages ({len(passages)})"

        data = load_company_data()
        revenue = data["revenue_quarterly"]
        passages = data["passages"]
        memory.put(
            "amd_revenue_quarterly", revenue, self.name,
            sources=[e["source"] for e in revenue], freshness="stored",
            description=f"{len(revenue)} quarters of revenue (fiscal labels)",
        )
        memory.put(
            "amd_passages", passages, self.name,
            sources=[p["source"] for p in passages], freshness="stored",
            description=f"{len(passages)} filing passages",
        )
        return (
            f"{self.name}: loaded sample_amd.json -> wrote "
            f"amd_revenue_quarterly ({len(revenue)} quarters), amd_passages ({len(passages)})"
        )
