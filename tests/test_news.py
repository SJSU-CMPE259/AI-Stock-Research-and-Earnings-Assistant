"""News tool + shared Finnhub client: fake responses only (CI never calls Finnhub)."""

from datetime import date

import pytest

from finsight.tools import finnhub_client, news

TODAY = date(2026, 10, 26)
ROWS = [  # made-up articles in Finnhub's format
    {"category": "company", "datetime": 1_761_400_000, "headline": "Nvidia unveils new chip",
     "id": 1, "related": "NVDA", "source": "Reuters", "summary": "Details...",
     "url": "https://example.com/a"},
    {"category": "company", "datetime": 1_761_300_000, "headline": "NVIDIA unveils new chip!",
     "id": 2, "related": "NVDA", "source": "Yahoo", "summary": "Same story, syndicated",
     "url": "https://example.com/b"},                                     # duplicate headline
    {"category": "company", "datetime": 1_761_200_000, "headline": "Nvidia supplier update",
     "id": 3, "related": "NVDA,TSM", "source": "CNBC", "summary": "",
     "url": "https://example.com/a"},                                     # duplicate URL
    {"category": "company", "datetime": 1_761_500_000, "headline": "Nvidia earnings preview",
     "id": 4, "related": "NVDA", "source": "MarketWatch", "summary": "Preview",
     "url": "https://example.com/c"},
    {"category": "company", "datetime": 1_761_100_000, "headline": "", "id": 5,
     "related": "NVDA", "source": "X", "summary": "", "url": "https://example.com/d"},  # no headline
]


def test_parse_sorts_dedupes_and_skips_empty():
    arts = news.parse_articles("nvda", ROWS)
    assert [a.article_id for a in arts] == [4, 1]           # newest first, 2 duplicates + 1 empty dropped
    assert arts[0].ticker == "NVDA" and arts[0].source == "MarketWatch"
    assert arts[0].published_at.endswith("+00:00")
    assert arts[1].citation().startswith("Reuters, 2025-10-25: Nvidia unveils new chip")


def test_related_split():
    rows = [dict(ROWS[2], url="https://example.com/z", headline="Other")]
    assert news.parse_articles("NVDA", rows)[0].related == ["NVDA", "TSM"]


def test_get_company_news_date_range_and_cache_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(finnhub_client, "get",
                        lambda path, params, ttl=0: calls.append((path, params, ttl)) or ROWS)
    arts = news.get_company_news("nvda", days=7, limit=1, today=TODAY)
    path, params, ttl = calls[0]
    assert path == "/company-news"
    assert params == {"symbol": "NVDA", "from": "2026-10-19", "to": "2026-10-26"}
    assert ttl == news.RECENT_TTL and len(arts) == 1

    news.get_company_news("NVDA", start="2026-08-20", end="2026-08-31", today=TODAY)
    assert calls[1][2] == news.PAST_TTL                       # past range -> long cache


def test_rejects_bad_ranges(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: [])
    with pytest.raises(ValueError, match="after end"):
        news.get_company_news("NVDA", start="2026-10-20", end="2026-10-10", today=TODAY)
    with pytest.raises(ValueError, match="too old"):
        news.get_company_news("NVDA", start="2024-01-01", end="2024-01-10", today=TODAY)


def test_finnhub_error_object(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: {"error": "Invalid symbol"})
    with pytest.raises(ValueError, match="Invalid symbol"):
        news.get_company_news("???", today=TODAY)


def test_get_news_for_keeps_going(monkeypatch):
    def fake(path, params, ttl=0):
        if params["symbol"] == "BAD":
            raise RuntimeError("boom")
        return ROWS
    monkeypatch.setattr(finnhub_client, "get", fake)
    out = news.get_news_for(["NVDA", "BAD"], limit=1)
    assert len(out["NVDA"]) == 1 and out["BAD"].startswith("RuntimeError")


def test_client_cache_hides_token_and_reuses(monkeypatch, tmp_path):
    """The shared client caches by path+params (no key in the file name) and reuses it."""
    monkeypatch.setenv("FINNHUB_API_KEY", "secret123")
    monkeypatch.setattr(finnhub_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finnhub_client, "MIN_INTERVAL", 0)
    hits = []

    class Resp:
        status_code = 200
        def json(self): return [{"ok": 1}]
        def raise_for_status(self): pass

    def fake_get(url, params, headers, timeout):
        hits.append((url, params, headers))
        return Resp()
    monkeypatch.setattr(finnhub_client._session, "get", fake_get)

    a = finnhub_client.get("/company-news", {"symbol": "NVDA"}, ttl=60)
    b = finnhub_client.get("/company-news", {"symbol": "NVDA"}, ttl=60)
    assert a == b == [{"ok": 1}] and len(hits) == 1             # second call from disk cache
    assert hits[0][2] == {"X-Finnhub-Token": "secret123"}       # key in header, not params
    assert "token" not in hits[0][1]
    assert all("secret123" not in p.read_text() + p.name for p in tmp_path.iterdir())


# --- relevance: the user's real NVDA result (Finnhub tagged unrelated Motley Fool pieces) ---
OFF_TOPIC = [
    {"datetime": 1_761_500_000 + i, "headline": h, "id": 100 + i, "related": "NVDA",
     "source": "Yahoo", "summary": s, "url": f"https://example.com/off{i}", "category": "company"}
    for i, (h, s) in enumerate([
        ("SpaceX's Adjusted Earnings Quadrupled in Q3. Is the Stock Heading Back to $200?", "SpaceX..."),
        ("Anthropic Just Filed to Go Public. Should You Buy the Stock?", "The AI lab..."),
        ("Why Range Resources Stock Surged Today", "Natural gas..."),
        ("Chip stocks rally", "Nvidia and AMD led gains."),            # named in summary -> medium
        ("Nvidia's new Blackwell orders", "Data center demand..."),      # named in headline -> high
        ("$NVDA hits record", "..."),                                    # cashtag -> high
    ])
]


def test_default_drops_articles_that_never_name_the_company(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: OFF_TOPIC)
    arts = news.get_company_news("NVDA", today=TODAY)
    assert [a.headline for a in arts] == ["$NVDA hits record", "Nvidia's new Blackwell orders",
                                          "Chip stocks rally"]
    assert [a.relevance for a in arts] == ["high", "high", "medium"]


def test_relevance_levels(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: OFF_TOPIC)
    assert len(news.get_company_news("NVDA", min_relevance="high", today=TODAY)) == 2
    assert len(news.get_company_news("NVDA", min_relevance="low", today=TODAY)) == 6
    with pytest.raises(ValueError):
        news.get_company_news("NVDA", min_relevance="very", today=TODAY)


def test_limit_applies_after_filtering(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: OFF_TOPIC)
    arts = news.get_company_news("NVDA", limit=2, today=TODAY)   # newest 3 are off-topic
    assert [a.relevance for a in arts] == ["high", "high"]


def test_whole_words_only():
    p = news.company_pattern("META")
    assert p.search("Meta beats estimates") and p.search("Facebook parent")
    assert not p.search("The metaverse is back") and not p.search("metadata leak")
    assert news.company_pattern("XYZ", names=["Acme Corp"]).search("acme corp wins")
