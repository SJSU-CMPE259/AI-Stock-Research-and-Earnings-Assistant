"""SEC EDGAR ticker/CIK lookup and filing retrieval."""
import logging
from dataclasses import dataclass
from typing import Optional

from backend.app.tools.sec_client import get_sec_client

logger = logging.getLogger(__name__)


@dataclass
class Filing:
    """A single SEC filing."""

    ticker: str
    cik: str
    form: str  # 10-K, 10-Q, 8-K, etc.
    accession: str  # With dashes: 0001047469-26-012345
    filing_date: str  # YYYY-MM-DD
    report_date: str  # YYYY-MM-DD
    url: str  # SEC full text URL
    primary_document: str  # e.g., 0001047469-26-012345-index.html


class Edgar:
    """SEC EDGAR data client."""

    def __init__(self):
        self.sec = get_sec_client()
        self._cik_cache = {}

    def get_cik(self, ticker: str) -> str:
        """Get CIK for a ticker from SEC company_tickers.json.

        Args:
            ticker: Stock ticker (e.g., NVDA, AAPL)

        Returns:
            CIK zero-padded to 10 digits (for data.sec.gov URLs)
        """
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        url = "https://www.sec.gov/files/company_tickers.json"
        data = self.sec.get(url)

        # company_tickers.json: {entry: {cik_str, ticker, title}, ...}
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                self._cik_cache[ticker] = cik
                logger.info(f"Lookup: {ticker} -> CIK {cik}")
                return cik

        raise ValueError(f"Ticker {ticker} not found in SEC EDGAR")

    def list_filings(
        self, ticker: str, form: str = "10-K", limit: int = 10
    ) -> list[Filing]:
        """List recent filings for a company.

        Args:
            ticker: Stock ticker
            form: Form type (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to return

        Returns:
            List of Filing objects, newest first
        """
        cik = self.get_cik(ticker)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = self.sec.get(url)

        filings = []
        recent = data.get("filings", {}).get("recent", {})

        # Parallel arrays in recent: form, accessionNumber, filingDate, reportDate, primaryDocument
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, f in enumerate(forms):
            if f == form and len(filings) < limit:
                accession = accessions[i]
                accession_no_dashes = accession.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0') or '0'}/"
                    f"{accession_no_dashes}/{primary_docs[i]}"
                )

                filing = Filing(
                    ticker=ticker,
                    cik=cik,
                    form=form,
                    accession=accession,
                    filing_date=filing_dates[i],
                    report_date=report_dates[i],
                    url=filing_url,
                    primary_document=primary_docs[i],
                )
                filings.append(filing)

        logger.info(f"Found {len(filings)} {form} filings for {ticker}")
        return filings

    def download_filing(self, filing: Filing, cache: bool = True) -> str:
        """Download a filing and return its HTML/text content.

        Args:
            filing: Filing object
            cache: Whether to use disk cache

        Returns:
            Filing HTML/text content
        """
        # Filing documents are HTML on SEC servers
        # For now, fetch the primary document (usually index.html or the 10-K/10-Q itself)
        logger.info(f"Downloading {filing.form} from {filing.accession}")

        try:
            response = self.sec.session.get(filing.url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to download {filing.url}: {e}")
            raise


def get_edgar() -> Edgar:
    """Get or create the Edgar client singleton."""
    if not hasattr(get_edgar, "_client"):
        get_edgar._client = Edgar()
    return get_edgar._client
