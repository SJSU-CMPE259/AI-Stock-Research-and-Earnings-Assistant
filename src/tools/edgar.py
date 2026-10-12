"""EDGAR tool: find a company's filings (10-K, 10-Q, 8-K) and fetch the documents.

EDGAR is the SEC's database of every filing a public company submits.
This module answers questions like "what is NVIDIA's latest 10-Q and where is it?"

Data sources (all free, no API key):
  - https://www.sec.gov/files/company_tickers.json          ticker -> CIK
  - https://data.sec.gov/submissions/CIK##########.json     list of filings
  - https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<doc>  the filing itself
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.tools.sec_client import get_json, get_text

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"


@dataclass
class Filing:
    ticker: str
    cik: str              # 10-digit, zero-padded
    form: str             # "10-Q", "10-K", "8-K", ...
    accession: str        # e.g. "0001045810-25-000230" — the filing's unique ID
    filing_date: str      # when it was filed with the SEC
    period_of_report: str # the period end date the filing covers
    primary_document: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_cik(ticker: str) -> str:
    """Return the 10-digit CIK for a ticker, e.g. 'NVDA' -> '0001045810'."""
    data = get_json(TICKERS_URL, ttl=7 * 24 * 60 * 60)
    t = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == t:
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found in SEC list: {ticker}")


def get_filings(ticker: str, form: str = "10-Q", limit: int = 4) -> list[Filing]:
    """Most recent filings of one form type, newest first.

    Uses the 'recent' block of the submissions API (about the last 1,000
    filings), which covers several years for most companies.
    """
    cik = get_cik(ticker)
    recent = get_json(SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"]
    out: list[Filing] = []
    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        acc = recent["accessionNumber"][i]
        doc = recent["primaryDocument"][i]
        out.append(
            Filing(
                ticker=ticker.upper(),
                cik=cik,
                form=f,
                accession=acc,
                filing_date=recent["filingDate"][i],
                period_of_report=recent["reportDate"][i],
                primary_document=doc,
                url=ARCHIVE_URL.format(
                    cik_int=int(cik), acc_nodash=acc.replace("-", ""), doc=doc
                ),
            )
        )
        if len(out) >= limit:
            break
    return out


def get_latest_filing(ticker: str, form: str = "10-Q") -> Filing:
    filings = get_filings(ticker, form, limit=1)
    if not filings:
        raise ValueError(f"No {form} found for {ticker}")
    return filings[0]


def get_filing_html(filing: Filing) -> str:
    """Raw HTML of the filing's main document (cleaning it is a Sprint 2 job)."""
    return get_text(filing.url)
