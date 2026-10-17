"""Quality checks on the database (spec section 5).

    python scripts/check_quality.py

Checks, per company:
  1. The four quarters add up to the full year (within 0.5%).
  2. No missing quarters in the last two years (no gaps between quarter periods).
  3. Every metric was found (missing ones may need a concept override).
Duplicate periods can't happen: the table has a UNIQUE constraint.
Prints PASS / WARN lines; exit code 1 if any check fails.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.datastore import db  # noqa: E402
from finsight.tools.xbrl import METRICS  # noqa: E402

SUM_METRICS = ["revenue", "net_income", "operating_income"]


def check(conn, today: date | None = None) -> list[tuple[str, str, str]]:
    today = today or date.today()
    two_years_ago = (today - timedelta(days=730)).isoformat()
    results = []  # (status, ticker, message)

    for (ticker,) in conn.execute("SELECT ticker FROM companies ORDER BY ticker"):
        # 1. quarters add up to the year
        for metric in SUM_METRICS:
            for fy, fy_value in conn.execute(
                "SELECT fiscal_year, value FROM financial_facts WHERE ticker=? AND metric=? "
                "AND period_type='annual'", (ticker, metric)):
                qs = conn.execute(
                    "SELECT value FROM financial_facts WHERE ticker=? AND metric=? "
                    "AND period_type='quarter' AND fiscal_year=?", (ticker, metric, fy)).fetchall()
                if len(qs) != 4:
                    continue  # incomplete year (e.g. the oldest one kept); covered by check 2
                total = sum(q[0] for q in qs)
                ok = abs(total - fy_value) <= 0.005 * abs(fy_value)
                results.append(("PASS" if ok else "FAIL", ticker,
                                f"{metric} FY{fy}: quarters {total:,.0f} vs year {fy_value:,.0f}"))

        # 2. no gaps between quarters in the last two years
        ends = conn.execute(
            "SELECT period_start, period_end, fiscal_year, fiscal_period FROM financial_facts "
            "WHERE ticker=? AND metric='revenue' AND period_type='quarter' AND period_end>=? "
            "ORDER BY period_end", (ticker, two_years_ago)).fetchall()
        gaps = [f"{b[3]} FY{b[2]}" for a, b in zip(ends, ends[1:])
                if (date.fromisoformat(b[0]) - date.fromisoformat(a[1])).days > 7]
        results.append(("FAIL" if gaps else "PASS", ticker,
                        f"revenue quarters, last 2 years: {len(ends)} found"
                        + (f"; gap before {', '.join(gaps)}" if gaps else "")))

        # 3. every metric found
        found = {m for (m,) in conn.execute(
            "SELECT DISTINCT metric FROM financial_facts WHERE ticker=?", (ticker,))}
        missing = [m for m in METRICS if m not in found]
        results.append(("WARN" if missing else "PASS", ticker,
                        "missing metrics: " + ", ".join(missing) if missing else "all metrics found"))
    return results


def main() -> int:
    conn = db.connect()
    results = check(conn)
    for status, ticker, msg in results:
        print(f"{status:<5} {ticker:<6} {msg}")
    fails = sum(1 for r in results if r[0] == "FAIL")
    print(f"\n{fails} failed, {sum(1 for r in results if r[0] == 'WARN')} warnings")
    print("Also do by hand: spot-check fiscal labels (NVDA, AAPL, MSFT) and compare "
          "3 companies' figures with their earnings press releases.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
