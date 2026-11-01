"""EDGAR tool: find a company's filings (10-K, 10-Q, 8-K) and fetch the documents.

EDGAR is the SEC's database of every filing a public company submits.
This module answers questions like "what is NVIDIA's latest 10-Q and where is it?"

Data sources (all free, no API key):
  - https://www.sec.gov/files/company_tickers.json          ticker -> CIK
  - https://data.sec.gov/submissions/CIK##########.json     list of filings
  - https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<doc>  the filing itself
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# Call through the module (sec_client.get_json) so tests can swap in fake SEC data.
from finsight.tools import sec_client

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

DEFAULT_FORMS = ("10-K", "10-Q", "8-K")

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
    items: list[str] = field(default_factory=list)  # 8-K item codes, e.g. ["2.02", "9.01"]

    @property
    def folder_url(self) -> str:
        """The filing's folder on sec.gov (all its documents and exhibits)."""
        return self.url.rsplit("/", 1)[0] + "/"

    @property
    def is_earnings_8k(self) -> bool:
        """8-K Item 2.02 = 'Results of Operations' = an earnings announcement."""
        return self.form == "8-K" and "2.02" in self.items    

    def to_dict(self) -> dict:
        return asdict(self)


def get_cik(ticker: str) -> str:
    """Return the 10-digit CIK for a ticker, e.g. 'NVDA' -> '0001045810'."""
    data =sec_client.get_json(TICKERS_URL, ttl=7 * 24 * 60 * 60)
    t = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == t:
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found in SEC list: {ticker}")


def get_submissions(cik: str) -> dict:
    """The company's submissions JSON: name, fiscal year end and its filings."""
    return sec_client.get_json(SUBMISSIONS_URL.format(cik=cik))


def get_company_info(ticker: str) -> dict:
    """ticker, cik, name, fiscal_year_end ("0125" = the fiscal year ends around Jan 25)."""
    cik = get_cik(ticker)
    sub = get_submissions(cik)
    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "name": sub.get("name"),
        "fiscal_year_end": sub.get("fiscalYearEnd"),
    }


def _make_filing(ticker: str, cik: str, row: dict) -> Filing:
    acc = row["accessionNumber"]
    doc = row.get("primaryDocument") or ""
    return Filing(
        ticker=ticker.upper(),
        cik=cik,
        form=row["form"],
        accession=acc,
        filing_date=row["filingDate"],
        period_of_report=row.get("reportDate") or "",
        primary_document=doc,
        url=ARCHIVE_URL.format(cik_int=int(cik), acc_nodash=acc.replace("-", ""), doc=doc),
        items=[x.strip() for x in (row.get("items") or "").split(",") if x.strip()],
    )


def _rows(block: dict):
    """The SEC stores filings as parallel lists; turn them into one dict per filing."""
    n = len(block.get("accessionNumber", []))
    for i in range(n):
        yield {k: v[i] for k, v in block.items() if isinstance(v, list) and len(v) == n}


def get_filings(ticker: str, form: str = "10-Q", limit: int = 4) -> list[Filing]:
    """Most recent filings of one form type, newest first.

    Uses the 'recent' block of the submissions API (about the last 1,000
    filings), which covers several years for most companies.
    """
    cik = get_cik(ticker)
    recent = get_submissions(cik)["filings"]["recent"]
    out: list[Filing] = []
    for row in _rows(recent):
        if row["form"] != form:
            continue
        out.append(_make_filing(ticker, cik, row))
        if len(out) >= limit:
            break
    return out


def list_filings_since(ticker: str, since: str, forms=DEFAULT_FORMS) -> list[Filing]:
    """All filings of the given forms filed on/after `since` (YYYY-MM-DD), newest first.

    Big companies file many insider-trading forms (Form 4), so the 'recent'
    block may not reach back 2-3 years. Older pages are read too when needed.
    """
    cik = get_cik(ticker)
    sub = get_submissions(cik)
    blocks = [sub["filings"]["recent"]]
    for page in sub["filings"].get("files", []):
        if page.get("filingTo", "9999") >= since:  # this page has filings in our window
            blocks.append(sec_client.get_json(SUBMISSIONS_PAGE_URL.format(name=page["name"])))

    out: dict[str, Filing] = {}
    for block in blocks:
        for row in _rows(block):
            if row.get("form") in forms and row.get("filingDate", "") >= since:
                out[row["accessionNumber"]] = _make_filing(ticker, cik, row)
    return sorted(out.values(), key=lambda f: f.filing_date, reverse=True)


def get_latest_filing(ticker: str, form: str = "10-Q") -> Filing:
    filings = get_filings(ticker, form, limit=1)
    if not filings:
        raise ValueError(f"No {form} found for {ticker}")
    return filings[0]


def get_filing_html(filing: Filing) -> str:
    """Raw HTML of the filing's main document (cleaning it is a Sprint 2 job)."""
    return sec_client.get_text(filing.url)


def download_document(filing: Filing, dest_dir: str | Path) -> Path:
    """Save the filing's main document as <dest_dir>/<TICKER>/<accession>.html."""
    path = Path(dest_dir) / filing.ticker / f"{filing.accession}.html"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(get_filing_html(filing))
    return path
