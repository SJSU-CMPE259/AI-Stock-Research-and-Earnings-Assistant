"""XBRL tool: get exact reported numbers (revenue, net income, ...) by fiscal period.

XBRL is the machine-readable version of the numbers in a filing. Every value
is tagged with a concept name (e.g. us-gaap:Revenues), a period, a unit and
the filing it came from. Using these instead of having the LLM read tables
is how FinSight hits its ">= 95% numeric accuracy" target.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Fiscal-quarter labels (the prototype's calendar-quarter bug):
  Each fact carries `fy` and `fp` (e.g. 2026, "Q3"), but those describe the
  FILING, not the fact. A 10-Q also repeats last year's numbers for
  comparison, and those carry the same fy/fp. So we only trust fy/fp for the
  fact whose period END matches the filing's own (latest) period end.
  We never use the `frame` field (e.g. CY2025Q3) for labels — that is calendar-based.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from src.finsight.tools.edgar import get_cik
from src.finsight.tools.sec_client import get_json

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Companies tag the same idea with different concepts. Try them in order.
METRICS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "total_assets": ["Assets"],          # balance-sheet item: point in time
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
}


@dataclass
class FactValue:
    """Store a financial fact with its reporting period and SEC filing details."""
    ticker: str
    metric: str
    concept: str
    fiscal_year: int
    fiscal_period: str   # "Q1", "Q2", "Q3", "FY"
    start: str | None    # None for point-in-time items like Assets
    end: str
    value: float
    unit: str
    form: str
    accession: str       # links the number back to the exact filing (for citations)
    filed: str

    @property
    def label(self) -> str:
        """Generate a readable fiscal period label (e.g., FY2026 Q3)."""
        return f"FY{self.fiscal_year} {self.fiscal_period}"

    def to_dict(self) -> dict:
        """Convert the financial fact to a dictionary with its period label."""
        d = asdict(self)
        d["label"] = self.label
        return d


def _days(start: str, end: str) -> int:
    """Calculate the number of days between two dates (YYYY-MM-DD)."""
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def get_company_facts(ticker: str) -> dict:
    """Retrieve a company's financial facts from the SEC XBRL API using its ticker."""
    return get_json(FACTS_URL.format(cik=get_cik(ticker)))


def extract_metric(facts: dict, ticker: str, metric: str,
                   period: str = "quarterly") -> list[FactValue]:
    """Extract quarterly or annual financial metrics from SEC XBRL data.

    Filters facts by reporting period and filing type, removes duplicates,
    and returns the results sorted from newest to oldest.

    This function makes no API calls, allowing unit testing with sample data.
    """
    # Get US GAAP financial facts and matching XBRL concepts
    gaap = facts.get("facts", {}).get("us-gaap", {})
    concepts = METRICS.get(metric, [metric])  # also accept a raw concept name

    for concept in concepts:
        if concept not in gaap:
            continue
        units = gaap[concept]["units"]
        unit = "USD" if "USD" in units else next(iter(units))
        rows = units[unit]

        # Latest period end in each filing = that filing's own period.
        latest_end: dict[str, str] = {}
        for r in rows:
            acc = r["accn"]
            if acc not in latest_end or r["end"] > latest_end[acc]:
                latest_end[acc] = r["end"]

        results: dict[tuple, FactValue] = {}
        for r in rows:
            form = r.get("form", "")
            if r["end"] != latest_end[r["accn"]]:
                continue  # comparative prior-period number; fy/fp would be wrong
            start = r.get("start")
            if start:  # duration item (income statement, cash flow)
                d = _days(start, r["end"])
                if period == "quarterly" and not (form.startswith("10-Q") and 80 <= d <= 100):
                    continue
                if period == "annual" and not (form.startswith("10-K") and 350 <= d <= 380):
                    continue
            else:  # point-in-time item (balance sheet)
                want = "10-Q" if period == "quarterly" else "10-K"
                if not form.startswith(want):
                    continue

            # Store the financial value with its SEC filing metadata            
            fv = FactValue(
                ticker=ticker.upper(), metric=metric, concept=concept,
                fiscal_year=int(r["fy"]), fiscal_period=r["fp"],
                start=start, end=r["end"], value=r["val"], unit=unit,
                form=form, accession=r["accn"], filed=r["filed"],
            )

            # Deduplicate periods, keeping the most recently filed value
            key = (fv.start, fv.end)
            # If a later amendment (10-Q/A) restates it, keep the newest filing.
            if key not in results or fv.filed > results[key].filed:
                results[key] = fv

        # Return results sorted by reporting period, newest first
        if results:
            return sorted(results.values(), key=lambda x: x.end, reverse=True)
    return []


def get_metric(ticker: str, metric: str, period: str = "quarterly",
               limit: int = 8) -> list[FactValue]:
    """Retrieve a company's financial metric from SEC XBRL data.

    Fetches company facts, extracts the requested quarterly or annual metric,
    and returns up to the specified number of results, newest first.
    """
    return extract_metric(get_company_facts(ticker), ticker, metric, period)[:limit]
