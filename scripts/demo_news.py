"""Recent company news (Finnhub free tier).

    python scripts/demo_news.py                  # all companies in config/companies.yaml, last 7 days
    python scripts/demo_news.py NVDA AAPL        # just these
    python scripts/demo_news.py NVDA --days 30 --limit 10
    python scripts/demo_news.py NVDA --relevance low     # everything Finnhub tagged
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import config  # noqa: E402
from finsight.tools.news import get_news_for  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tickers", nargs="*")
ap.add_argument("--days", type=int, default=7)
ap.add_argument("--limit", type=int, default=5, help="articles per company")
ap.add_argument("--relevance", default="medium", choices=["high", "medium", "low"],
                help="high: company in headline; medium: headline or summary; low: all")
args = ap.parse_args()

tickers = [t.upper() for t in args.tickers] or config.load_companies()
print(f"News for {len(tickers)} companies, last {args.days} days (about 1 call per second)...\n")
for ticker, articles in get_news_for(tickers, days=args.days, limit=args.limit,
                                         min_relevance=args.relevance).items():
    if isinstance(articles, str):
        print(f"{ticker}: {articles}\n")
        continue
    print(f"{ticker}: {len(articles)} articles")
    for a in articles:
        print(f"  {a.date}  {a.relevance:<6} {a.source:<14.14} {a.headline[:85]}")
    print()
