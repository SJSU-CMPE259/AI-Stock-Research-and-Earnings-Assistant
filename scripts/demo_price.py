"""Live prices for your companies (Finnhub free tier).

    python scripts/price_demo.py                 # all companies in config/companies.yaml
    python scripts/price_demo.py NVDA AAPL       # just these
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import config  # noqa: E402
from finsight.tools.prices import Quote, get_quotes  # noqa: E402

tickers = [t.upper() for t in sys.argv[1:]] or config.load_companies()
print(f"Fetching {len(tickers)} quotes (about 1 per second to respect the free limit)...\n")
for ticker, q in get_quotes(tickers).items():
    print("  " + (q.summary() if isinstance(q, Quote) else f"{ticker}: {q}"))
