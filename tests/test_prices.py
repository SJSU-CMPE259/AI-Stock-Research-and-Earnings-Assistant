"""Live price tool: tested with fake Finnhub responses (CI never calls Finnhub)."""

import pytest

from finsight.tools import prices

SAMPLE = {"c": 182.4, "d": 2.25, "dp": 1.2489, "h": 184.1, "l": 179.8,
          "o": 180.5, "pc": 180.15, "t": 1_760_644_800}  # made-up values


def test_parse_quote_fields():
    q = prices.parse_quote("nvda", SAMPLE, now=SAMPLE["t"] + 60)
    assert (q.ticker, q.price, q.prev_close, q.change_pct) == ("NVDA", 182.4, 180.15, 1.2489)
    assert q.as_of == "2025-10-16T20:00:00+00:00"
    assert q.market_open_likely is True
    assert q.summary().startswith("NVDA $182.40 (+1.25%")


def test_old_price_means_market_closed():
    q = prices.parse_quote("NVDA", SAMPLE, now=SAMPLE["t"] + 3 * 24 * 3600)  # a weekend later
    assert q.market_open_likely is False
    assert "market closed" in q.summary()


def test_unknown_ticker_raises_instead_of_zero_price():
    zeros = {"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}
    with pytest.raises(ValueError):
        prices.parse_quote("NOTAREALTICKER", zeros)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        prices._api_key()


def test_get_quote_uses_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(prices, "_get", lambda params: calls.append(params) or SAMPLE)
    prices._cache.clear()
    a = prices.get_quote("NVDA")
    b = prices.get_quote("nvda")          # within 60 s -> from cache
    assert a is b and len(calls) == 1
    prices.get_quote("NVDA", max_age=0)   # force a fresh call
    assert len(calls) == 2


def test_get_quotes_keeps_going_after_a_bad_ticker(monkeypatch):
    def fake(params):
        return SAMPLE if params["symbol"] == "NVDA" else {"c": 0, "t": 0}
    monkeypatch.setattr(prices, "_get", fake)
    prices._cache.clear()
    out = prices.get_quotes(["NVDA", "BAD"])
    assert out["NVDA"].price == 182.4
    assert out["BAD"].startswith("ValueError")
