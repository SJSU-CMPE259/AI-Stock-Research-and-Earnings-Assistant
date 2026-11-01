"""XBRL tool: get exact reported numbers (revenue, net income, ...) by fiscal period.

XBRL is the machine-readable version of the numbers in a filing. Every value
is tagged with a concept name (e.g. us-gaap:Revenues), a period, a unit and
the filing it came from. Using these instead of having the LLM read tables
is how FinSight hits its ">= 95% numeric accuracy" target.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Cleaning rules (SEC data workstream spec, D2):
  1. Fiscal labels come from the filing. Each fact carries `fy` and `fp`
     (e.g. 2026, "Q3"), but those describe the FILING, not the fact. A 10-Q
     also repeats last year's numbers for comparison, and those carry the same
     fy/fp. So we only trust fy/fp for the fact whose period END matches the
     filing's own (latest) period end. We never use `frame` (e.g. CY2025Q3):
     that is calendar-based.
  2. Quarters are three-month figures; year-to-date figures are dropped
     (the nine-month figure is only used to derive Q4).
  3. If a period was reported more than once (e.g. a 10-Q/A), keep the value
     from the filing that FIRST reported it.
  4. Q4 = full year - nine months, marked derived=True. Not for per-share values (EPS).
  5. Balance-sheet items (cash, assets) are snapshots on one date: period_type "instant".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

from finsight.tools import sec_client
from finsight.tools.edgar import get_cik

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FOLDER_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"


# Companies tag the same idea with different concepts. Try them in order.
# Per-company fixes go in config/concepts.yaml (see datastore/ingest.py).
METRICS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
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
    period_type: str = ""     # "quarter", "annual" or "instant"
    derived: bool = False     # True for Q4 = full year - nine months
    cik: str = ""

    @property
    def label(self) -> str:
        """Readable fiscal period label in the spec's format: 'Q2 FY2027' or 'FY2026'."""
        return fiscal_label(self.fiscal_year, self.fiscal_period)

    @property
    def source_url(self) -> str:
        """The filing's folder on sec.gov."""
        return folder_url(self.cik, self.accession) if self.cik else ""

    def to_dict(self) -> dict:
        """Convert the financial fact to a dictionary with its period label."""
        d = asdict(self)
        d["label"] = self.label
        d["source_url"] = self.source_url
        return d


def fiscal_label(fy: int, fp: str) -> str:
    """(2027, "Q2") -> "Q2 FY2027";  (2026, "FY") -> "FY2026"."""
    return f"FY{fy}" if fp == "FY" else f"{fp} FY{fy}"


def folder_url(cik: str, accession: str) -> str:
    return FOLDER_URL.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""))


def _days(start: str, end: str) -> int:
    """Calculate the number of days between two dates (YYYY-MM-DD)."""
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def get_company_facts(ticker: str) -> dict:
    """Retrieve a company's financial facts from the SEC XBRL API using its ticker."""
    return sec_client.get_json(FACTS_URL.format(cik=get_cik(ticker)))


def _own_period_rows(rows: list[dict]) -> list[dict]:
    """Keep only rows whose end date is the latest end date within their filing (rule 1)."""
    latest: dict[str, str] = {}
    for r in rows:
        if r["end"] > latest.get(r["accn"], ""):
            latest[r["accn"]] = r["end"]
    return [r for r in rows if r["end"] == latest[r["accn"]] and r.get("fy") and r.get("fp")]


def _period_type(r: dict) -> str | None:
    """quarter / ytd9 / annual / instant, or None if we don't keep this row."""
    form = r.get("form", "")
    if not form.startswith(("10-Q", "10-K")):
        return None
    if "start" not in r:
        return "instant"
    d = _days(r["start"], r["end"])
    if form.startswith("10-Q") and 80 <= d <= 100:
        return "quarter"
    if form.startswith("10-Q") and r.get("fp") == "Q3" and 250 <= d <= 290:
        return "ytd9"   # nine months year-to-date: only used to derive Q4
    if form.startswith("10-K") and 350 <= d <= 380:
        return "annual"
    return None         # other year-to-date figures (six months, ...)


def clean_metric(facts: dict, ticker: str, metric: str,
                 concepts: list[str] | None = None) -> list[FactValue]:
    """All clean values of one metric: quarters (incl. derived Q4), years and snapshots.

    No API calls, so it can be unit-tested with sample data.
    `concepts` overrides METRICS[metric] for one company.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    concepts = concepts or METRICS.get(metric, [metric])  # also accept a raw concept name
    cik = str(facts.get("cik", "")).zfill(10) if facts.get("cik") else ""

    # key (period_type, start, end) -> (priority, row, concept, unit)
    chosen: dict[tuple, tuple[int, dict, str, str]] = {}
    for priority, concept in enumerate(concepts):
        if concept not in gaap:
            continue
        units = gaap[concept]["units"]
        unit = next((u for u in ("USD", "USD/shares") if u in units), next(iter(units)))
        for r in _own_period_rows(units[unit]):
            ptype = _period_type(r)
            if not ptype:
                continue
            key = (ptype, r.get("start"), r["end"])
            old = chosen.get(key)
            # earlier concept in the list wins; within a concept, the FIRST filed wins (rule 3)
            if old is None or (priority, r["filed"]) < (old[0], old[1]["filed"]):
                chosen[key] = (priority, r, concept, unit)

    out: list[FactValue] = []
    nine_months: dict[str, dict] = {}   # fiscal-year start -> nine-month row
    for (ptype, _start, _end), (_p, r, concept, unit) in chosen.items():
        if ptype == "ytd9":
            nine_months[r["start"]] = r
            continue
        out.append(FactValue(
            ticker=ticker.upper(), metric=metric, concept=concept,
            fiscal_year=int(r["fy"]), fiscal_period="FY" if ptype == "annual" else r["fp"],
            start=r.get("start"), end=r["end"], value=r["val"], unit=unit,
            form=r["form"], accession=r["accn"], filed=r["filed"],
            period_type=ptype, cik=cik,
        ))

    # Rule 4: Q4 = full year - nine months (not for per-share values like EPS)
    quarter_ends = {f.end for f in out if f.period_type == "quarter"}
    for year in [f for f in out if f.period_type == "annual" and "/shares" not in f.unit]:
        nine = nine_months.get(year.start)
        if not nine or year.end in quarter_ends:
            continue
        out.append(FactValue(
            ticker=year.ticker, metric=metric, concept=year.concept,
            fiscal_year=year.fiscal_year, fiscal_period="Q4",
            start=(date.fromisoformat(nine["end"]) + timedelta(days=1)).isoformat(),
            end=year.end, value=year.value - nine["val"], unit=year.unit,
            form=year.form, accession=year.accession, filed=year.filed,
            period_type="quarter", derived=True, cik=cik,
        ))

    return sorted(out, key=lambda f: (f.end, f.period_type))


def extract_metric(facts: dict, ticker: str, metric: str,
                   period: str = "quarterly") -> list[FactValue]:
    """Extract quarterly or annual values of one metric, newest first.

    period="quarterly": three-month values (Q1-Q3 from 10-Qs, Q4 derived),
                        or 10-Q snapshots for balance-sheet items.
    period="annual":    full-year values from 10-Ks, or 10-K snapshots.
    This function makes no API calls, allowing unit testing with sample data.
    """
    rows = clean_metric(facts, ticker, metric)
    if period == "quarterly":
        keep = [f for f in rows if f.period_type == "quarter"
                or (f.period_type == "instant" and f.form.startswith("10-Q"))]
    else:
        keep = [f for f in rows if f.period_type == "annual"
                or (f.period_type == "instant" and f.form.startswith("10-K"))]
    return sorted(keep, key=lambda x: x.end, reverse=True)


def extract_all(facts: dict, ticker: str, overrides: dict | None = None) -> list[FactValue]:
    """Clean values for every metric in METRICS (used by the database loader)."""
    overrides = overrides or {}
    out: list[FactValue] = []
    for metric in METRICS:
        out.extend(clean_metric(facts, ticker, metric, overrides.get(metric)))
    return out


def filing_periods(facts: dict) -> dict[str, tuple[int, str, str]]:
    """accession -> (fiscal_year, fiscal_period, own period end) for every 10-K / 10-Q.

    Used to put fiscal labels on the filings list.
    """
    out: dict[str, tuple[int, str, str]] = {}
    for concept in facts.get("facts", {}).get("us-gaap", {}).values():
        for rows in concept["units"].values():
            for r in rows:
                if not r.get("form", "").startswith(("10-Q", "10-K")) or not r.get("fy"):
                    continue
                cur = out.get(r["accn"])
                if cur is None or r["end"] > cur[2]:
                    out[r["accn"]] = (int(r["fy"]), r["fp"], r["end"])
    return out


def get_metric(ticker: str, metric: str, period: str = "quarterly",
               limit: int = 8) -> list[FactValue]:
    """Retrieve a company's financial metric LIVE from the SEC (quick checks, notebooks).

    The rest of FinSight should use finsight.datastore.api.get_metric, which
    reads our database instead and adds the % changes.
    """
    return extract_metric(get_company_facts(ticker), ticker, metric, period)[:limit]
