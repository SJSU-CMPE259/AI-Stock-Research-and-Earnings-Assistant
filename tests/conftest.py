"""Shared test setup: fake SEC responses from tests/fixtures, so CI never calls the SEC."""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FIX = Path(__file__).parent / "fixtures"
TODAY = date(2026, 10, 15)


def load(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def companyfacts():
    return load("nvda_companyfacts.json")


@pytest.fixture
def fake_sec(monkeypatch, tmp_path):
    """Route all SEC calls to the fixture files; put data/ in a temp folder."""
    from finsight import config
    from finsight.tools import sec_client

    def fake_get_json(url, ttl=0):
        if url.endswith("company_tickers.json"):
            return load("company_tickers.json")
        if "submissions-001" in url:
            return load("nvda_submissions_001.json")
        if "/submissions/" in url:
            return load("nvda_submissions.json")
        if "/companyfacts/" in url:
            return load("nvda_companyfacts.json")
        raise AssertionError(f"unexpected URL {url}")

    def fake_get_text(url, ttl=0):
        return ("<html><head><title>x</title></head><body><ix:header>hidden xbrl</ix:header>"
                f"<p>Item 2. Management&#8217;s Discussion</p><p>Revenue grew.</p>"
                f"<!-- {url} --></body></html>")

    monkeypatch.setattr(sec_client, "get_json", fake_get_json)
    monkeypatch.setattr(sec_client, "get_text", fake_get_text)
    monkeypatch.setattr(config, "FILINGS_DIR", tmp_path / "filings")
    return tmp_path


@pytest.fixture
def built_db(fake_sec):
    """A database built from the fixtures by the real loader."""
    from finsight.datastore import db, ingest

    conn = db.connect(fake_sec / "test.db")
    ingest.ingest_company(conn, "NVDA", years=3, today=TODAY, run_started="2026-10-15T00:00:00")
    return conn
