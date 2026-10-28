"""Company metadata + logo helper, shared across the UI."""
from __future__ import annotations

# Tickers the picker offers. Filing Q&A and the real_time route work for any of them;
# numbers/sentiment have bundled sample data for AMD only (UI flags this).
COMPANY_INFO: dict[str, dict] = {
    "NVDA": {"name": "NVIDIA Corp."},
    "AMD": {"name": "Advanced Micro Devices"},
    "AAPL": {"name": "Apple Inc."},
    "MSFT": {"name": "Microsoft Corp."},
    "TSLA": {"name": "Tesla, Inc."},
    "GOOGL": {"name": "Alphabet Inc."},
    "AMZN": {"name": "Amazon.com, Inc."},
    "META": {"name": "Meta Platforms, Inc."},
    "INTC": {"name": "Intel Corp."},
    "JPM": {"name": "JPMorgan Chase & Co."},
}

SAMPLE_TICKER = "AMD"  # the only company with bundled numbers/sentiment sample data


def company_name(ticker: str) -> str:
    return COMPANY_INFO.get(ticker.upper(), {}).get("name", ticker.upper())


def logo_url(ticker: str) -> str:
    """Stock logo by ticker (FinancialModelingPrep, works for any US ticker)."""
    return f"https://financialmodelingprep.com/image-stock/{ticker.upper()}.png"
