"""Company metadata + logo helper, shared across the UI."""
from __future__ import annotations

# Tickers the picker offers. Filing Q&A and the real_time route work for any of them;
# numbers/sentiment have bundled sample data for AMD only (UI flags this).
COMPANY_INFO: dict[str, dict] = {
    "NVDA": {"name": "NVIDIA Corp.", "short": "NVIDIA"},
    "AMD": {"name": "Advanced Micro Devices", "short": "AMD"},
    "AAPL": {"name": "Apple Inc.", "short": "Apple"},
    "MSFT": {"name": "Microsoft Corp.", "short": "Microsoft"},
    "TSLA": {"name": "Tesla, Inc.", "short": "Tesla"},
    "GOOGL": {"name": "Alphabet Inc.", "short": "Alphabet"},
    "AMZN": {"name": "Amazon.com, Inc.", "short": "Amazon"},
    "META": {"name": "Meta Platforms, Inc.", "short": "Meta"},
    "INTC": {"name": "Intel Corp.", "short": "Intel"},
    "JPM": {"name": "JPMorgan Chase & Co.", "short": "JPMorgan Chase"},
}

SAMPLE_TICKER = "AMD"  # the only company with bundled numbers/sentiment sample data


def company_name(ticker: str) -> str:
    return COMPANY_INFO.get(ticker.upper(), {}).get("name", ticker.upper())


def short_name(ticker: str) -> str:
    """Everyday name for headings ("NVIDIA", not "NVIDIA Corp.")."""
    return COMPANY_INFO.get(ticker.upper(), {}).get("short", ticker.upper())


def logo_url(ticker: str) -> str:
    """Stock logo by ticker (FinancialModelingPrep, works for any US ticker)."""
    return f"https://financialmodelingprep.com/image-stock/{ticker.upper()}.png"
