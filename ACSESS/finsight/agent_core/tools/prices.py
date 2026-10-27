"""Live stock price tool (Finnhub free tier).

Answers "what is NVIDIA trading at right now?" for the agents and the UI.

Source: https://finnhub.io/api/v1/quote?symbol=NVDA   (free key from finnhub.io)
  c  = current price      d  = change        dp = change %
  h  = day high           l  = day low       o  = open
  pc = previous close     t  = time of the last price (Unix seconds)

Rules:
  - The key comes from .env (FINNHUB_API_KEY). Never commit it.
  - Free tier is about 60 calls/minute, shared with news.py through
    finnhub_client.py (>= 1.1 s between calls). Each quote is reused for 60 s.
  - Prices are NOT stored in the SQLite datastore: they change every second.
    Filings and reported figures (SEC) stay the "batch path"; prices are live.
  - Unknown tickers come back from Finnhub as all zeros; we raise an error
    instead of reporting a price of $0.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Shared with news.py: one key, one rate limiter for the 60 calls/minute budget.
from agent_core.tools import finnhub_client

QUOTE_TTL = 60            # seconds a quote is reused before asking again
STALE_AFTER = 15 * 60     # last trade older than this -> market likely closed

_cache: dict[str, tuple[float, "Quote"]] = {}


@dataclass
class Quote:
    ticker: str
    price: float              # last price
    change: float             # vs previous close, in $
    change_pct: float         # vs previous close, in %
    open: float
    high: float
    low: float
    prev_close: float
    as_of: str               # time of the last price, ISO UTC
    market_open_likely: bool  # False if the last price is old (weekend, night)
    source: str = "Finnhub"
    source_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One line for chat answers, e.g. 'NVDA $182.40 (+1.25%, as of 2026-10-16 20:00 UTC, Finnhub)'."""
        when = self.as_of.replace("T", " ")[:16] + " UTC"
        status = "" if self.market_open_likely else ", market closed"
        return (f"{self.ticker} ${self.price:,.2f} ({self.change_pct:+.2f}%, "
                f"as of {when}{status}, {self.source})")


def _get(params: dict) -> dict:
    """GET /quote through the shared Finnhub client (rate limit + retries live there)."""
    return finnhub_client.get("/quote", params)


def parse_quote(ticker: str, data: dict, now: float | None = None) -> Quote:
    """Turn Finnhub's short field names into a Quote. Pure function (unit-tested)."""
    if not data or not data.get("t") or not data.get("c"):
        raise ValueError(f"No price for {ticker!r}: unknown ticker or not a US stock")
    now = now if now is not None else time.time()
    return Quote(
        ticker=ticker.upper(),
        price=float(data["c"]),
        change=float(data.get("d") or 0.0),
        change_pct=float(data.get("dp") or 0.0),
        open=float(data.get("o") or 0.0),
        high=float(data.get("h") or 0.0),
        low=float(data.get("l") or 0.0),
        prev_close=float(data.get("pc") or 0.0),
        as_of=datetime.fromtimestamp(data["t"], tz=timezone.utc).isoformat(timespec="seconds"),
        market_open_likely=(now - data["t"]) < STALE_AFTER,
        source_url=f"https://finnhub.io/quote/{ticker.upper()}",
    )


def get_quote(ticker: str, max_age: int = QUOTE_TTL) -> Quote:
    """Latest price for one ticker, e.g. get_quote("NVDA").price.

    Reuses a quote fetched in the last `max_age` seconds (0 = always ask Finnhub).
    """
    t = ticker.upper().strip()
    hit = _cache.get(t)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    quote = parse_quote(t, _get({"symbol": t}))
    _cache[t] = (time.time(), quote)
    return quote


def get_quotes(tickers: list[str], max_age: int = QUOTE_TTL) -> dict[str, Quote | str]:
    """Quotes for several tickers. A failed ticker gets an error message instead of a Quote."""
    out: dict[str, Quote | str] = {}
    for t in tickers:
        try:
            out[t.upper()] = get_quote(t, max_age)
        except Exception as e:  # noqa: BLE001 — one bad ticker shouldn't stop the rest
            out[t.upper()] = f"{type(e).__name__}: {e}"
    return out
