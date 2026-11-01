"""Build the FinSight database from the SEC in one command.

Run from the repo root:
    python scripts/build_db.py                  # all companies in config/companies.yaml
    python scripts/build_db.py --fresh          # delete the database and rebuild from scratch
    python scripts/build_db.py --tickers NVDA AAPL --no-docs   # quick test run

Running it again later just adds what's new (that's also what the D4 job will do).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import config  # noqa: E402
from finsight.datastore import db, ingest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", help="default: config/companies.yaml")
    ap.add_argument("--years", type=int, default=3, help="how many years of filings (default 3)")
    ap.add_argument("--no-docs", action="store_true", help="skip downloading filing documents")
    ap.add_argument("--fresh", action="store_true", help="delete the database first")
    ap.add_argument("--db", default=str(config.DB_PATH), help=f"database file (default {config.DB_PATH})")
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or config.load_companies())]
    if not tickers:
        print("No tickers: add them to config/companies.yaml or pass --tickers")
        return 1
    if args.fresh:
        ingest.reset_database(Path(args.db))

    conn = db.connect(args.db)
    print(f"Loading {len(tickers)} companies into {args.db} ...")
    results = ingest.ingest_all(conn, tickers, years=args.years, download_docs=not args.no_docs)

    failed = 0
    for r in results:
        if "error" in r:
            failed += 1
            print(f"  ✗ {r['ticker']:<6} {r['error']}")
        else:
            print(f"  ✓ {r['ticker']:<6} {r['new_filings']:>4} new filings   {r['new_facts']:>4} new figures")

    totals = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("companies", "filings", "financial_facts")}
    print(f"\nDatabase now has: {totals}")
    print("Next: python scripts/check_quality.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
