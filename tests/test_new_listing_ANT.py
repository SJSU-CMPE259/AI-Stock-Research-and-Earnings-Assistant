"""Tests for adding a newly listed company (ANT, Anthropic): no network, no API keys.

    pytest tests/test_new_listing.py -v

Covers the four changes on branch p1/add-ANT:
  1. edgar.get_cik      refetches the SEC ticker list on a miss, then uses CIK_FALLBACK
  2. ingest             loads a company whose XBRL companyfacts is 404 (no 10-Q/10-K yet)
  3. collector          a company's FIRST 10-Q/10-K raises an event even with zero filings before
  4. news               "ANT" matches in capitals only; "Anthropic"/"Claude" in any case
"""

from __future__ import annotations

import requests
import pytest

from finsight.datastore import collector, db, ingest
from finsight.tools import edgar, news, xbrl


# ---------------------------------------------------------------- 1. get_cik

def _fake_ticker_list(fresh_has_ant: bool):
    """Fake sec_client.get_json: the cached copy (ttl > 0) predates the IPO."""
    calls = []
    old = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
    new = {**old, "1": {"cik_str": 2130357, "ticker": "ANT", "title": "Anthropic, PBC"}}

    def get_json(url, ttl=0):
        calls.append(ttl)
        return new if (ttl == 0 and fresh_has_ant) else old
    return get_json, calls


def test_get_cik_known_ticker_uses_cache_only(monkeypatch):
    fake, calls = _fake_ticker_list(fresh_has_ant=True)
    monkeypatch.setattr(edgar.sec_client, "get_json", fake)
    assert edgar.get_cik("nvda") == "0001045810"
    assert calls == [7 * 24 * 60 * 60]            # no extra SEC call


def test_get_cik_refetches_when_cache_is_stale(monkeypatch):
    fake, calls = _fake_ticker_list(fresh_has_ant=True)
    monkeypatch.setattr(edgar.sec_client, "get_json", fake)
    assert edgar.get_cik("ANT") == "0002130357"
    assert calls == [7 * 24 * 60 * 60, 0]         # cached, then fresh


def test_get_cik_fallback_when_sec_list_lacks_it(monkeypatch):
    fake, _ = _fake_ticker_list(fresh_has_ant=False)
    monkeypatch.setattr(edgar.sec_client, "get_json", fake)
    assert edgar.get_cik("ANT") == edgar.CIK_FALLBACK["ANT"]


def test_get_cik_unknown_ticker_still_raises(monkeypatch):
    fake, _ = _fake_ticker_list(fresh_has_ant=False)
    monkeypatch.setattr(edgar.sec_client, "get_json", fake)
    with pytest.raises(ValueError, match="not found"):
        edgar.get_cik("ZZZZ")


# ---------------------------------------------------------------- 2. ingest

def _http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=resp)


def _filing(form="8-K", acc="0002130357-26-000010", date="2026-11-19", items=()):
    return edgar.Filing(ticker="ANT", cik="0002130357", form=form, accession=acc,
                        filing_date=date, period_of_report=date, primary_document="d.htm",
                        url=f"https://www.sec.gov/Archives/edgar/data/2130357/{acc}/d.htm",
                        items=list(items))


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "test.db")


@pytest.fixture
def fake_sec(monkeypatch):
    """ANT on the SEC: company info, one 8-K, and NO companyfacts (404)."""
    monkeypatch.setattr(edgar, "get_company_info", lambda t: {
        "ticker": "ANT", "cik": "0002130357", "name": "Anthropic, PBC", "fiscal_year_end": "1231"})
    filings = [_filing()]
    monkeypatch.setattr(edgar, "list_filings_since", lambda t, since, forms=None: list(filings))

    def no_facts(t):
        raise _http_error(404)
    monkeypatch.setattr(xbrl, "get_company_facts", no_facts)
    return filings          # tests can append to simulate a new filing


def test_ingest_new_listing_without_xbrl(conn, fake_sec):
    stats = ingest.ingest_company(conn, "ANT", download_docs=False)
    assert stats == {"ticker": "ANT", "new_filings": 1, "new_facts": 0}
    assert conn.execute("SELECT name FROM companies WHERE ticker='ANT'").fetchone()[0] == "Anthropic, PBC"
    assert conn.execute("SELECT COUNT(*) FROM filings WHERE ticker='ANT'").fetchone()[0] == 1


def test_ingest_other_http_errors_still_fail(conn, fake_sec, monkeypatch):
    def server_error(t):
        raise _http_error(503)
    monkeypatch.setattr(xbrl, "get_company_facts", server_error)
    with pytest.raises(requests.HTTPError):
        ingest.ingest_company(conn, "ANT", download_docs=False)


# ---------------------------------------------------------------- 3. collector

def test_first_run_is_baseline_no_events(conn, fake_sec):
    r = collector.collect_company(conn, "ANT", download_docs=False)
    assert r["baseline"] is True and r["events"] == 0
    assert len(r["new_filings"]) == 1


def test_baseline_loads_company_even_with_zero_filings(conn, fake_sec):
    fake_sec.clear()                                  # IPO day: no 10-K/10-Q/8-K at all
    r = collector.collect_company(conn, "ANT", download_docs=False)
    assert r["baseline"] is True
    assert conn.execute("SELECT 1 FROM companies WHERE ticker='ANT'").fetchone()


def test_first_10q_after_ipo_raises_event(conn, fake_sec):
    """The bug this branch fixes: with no filings stored, the first 10-Q looked like a baseline."""
    fake_sec.clear()
    collector.collect_company(conn, "ANT", download_docs=False)       # IPO day: nothing yet

    fake_sec.append(_filing(form="10-Q", acc="0002130357-27-000001", date="2027-01-04"))
    r = collector.collect_company(conn, "ANT", download_docs=False)
    assert r["baseline"] is False
    assert r["events"] == 1
    ev = conn.execute("SELECT event_type, form FROM filing_events").fetchone()
    assert tuple(ev) == ("quarterly_report", "10-Q")


def test_nothing_new_is_cheap(conn, fake_sec, monkeypatch):
    collector.collect_company(conn, "ANT", download_docs=False)
    called = []
    monkeypatch.setattr(ingest, "ingest_company", lambda *a, **k: called.append(1))
    r = collector.collect_company(conn, "ANT", download_docs=False)
    assert r["new_filings"] == [] and called == []    # loader not run again


# ---------------------------------------------------------------- 4. news

@pytest.mark.parametrize("text, hit", [
    ("ANT jumps 20% in Nasdaq debut", True),
    ("Buy $ANT?", True),
    ("Anthropic prices IPO at $83", True),
    ("anthropic shares slip", True),
    ("New Claude model released", True),
    ("Ant Group files for Hong Kong listing", False),
    ("an ant colony study", False),
    ("Antenna maker beats estimates", False),
    ("Anthropics of the market", False),
])
def test_ant_news_matching(text, hit):
    assert bool(news.company_pattern("ANT").search(text)) is hit


def test_other_companies_unchanged():
    meta = news.company_pattern("META")
    assert meta.search("Meta shares rise") and meta.search("META earnings")
    assert not meta.search("the metaverse")
    assert news.company_pattern("NVDA").search("Nvidia beats")


def test_relevance_levels_for_ant():
    p = news.company_pattern("ANT")
    assert news.relevance("Anthropic beats estimates", "", p) == "high"
    assert news.relevance("AI stocks rally", "Gains led by Anthropic.", p) == "medium"
    assert news.relevance("Ant Group update", "Ant Group shares rose.", p) == "low"
