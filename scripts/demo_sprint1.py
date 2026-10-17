"""Week 5 demo (stored path): a company's latest 10-Q and its figures, from OUR database.

    python scripts/build_db.py --tickers NVDA     # once, to load the data
    python scripts/demo_sprint1.py NVDA
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight.datastore import api  # noqa: E402

ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()

company = api.get_company(ticker)
if not company:
    sys.exit(f"{ticker} is not in the database. Run: python scripts/build_db.py --tickers {ticker}")
print(f"\n{company['name']} ({ticker})  CIK {company['cik']}  fiscal year ends {company['fiscal_year_end']}")

for f in api.list_filings(ticker, "10-Q", limit=1):
    print(f"Latest 10-Q: {f['fiscal_label']}  period ending {f['period_end']}  filed {f['filing_date']}")
    print(f"  {f['document_url']}\n")


def pct(x):
    return "    —" if x is None else f"{x:+6.1f}%"


for metric in ["revenue", "net_income", "eps_diluted"]:
    print(metric)
    for r in api.get_metric(ticker, metric, "quarter", n=5):
        v = f"{r['value'] / 1e9:>8.2f}B" if r["unit"] == "USD" else f"{r['value']:>9.2f}"
        flag = " (derived)" if r["derived"] else ""
        print(f"  {r['fiscal_label']:<10} {v}   QoQ {pct(r['change_vs_prior_quarter_pct'])}"
              f"   YoY {pct(r['change_vs_prior_year_pct'])}   [{r['form']} {r['accession']}]{flag}")
    print()
