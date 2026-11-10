"""D4 collection job: detect new filings, store them, raise events (no live SEC calls)."""

import copy
import os
import time

import pytest
from conftest import TODAY, load

from finsight.datastore import api, collector, db
from finsight.tools import sec_client

NEW_FILINGS = [  # appear on the SEC "after" the first run
    {"accessionNumber": "0001045810-26-000900", "filingDate": "2026-10-14", "reportDate": "2026-10-14",
     "form": "8-K", "primaryDocument": "nvda-8k-earnings.htm", "items": "2.02,9.01"},
    {"accessionNumber": "0001045810-26-000901", "filingDate": "2026-10-13", "reportDate": "2026-10-13",
     "form": "8-K", "primaryDocument": "nvda-8k-officer.htm", "items": "5.02"},
    {"accessionNumber": "0001045810-26-000902", "filingDate": "2026-10-13", "reportDate": "",
     "form": "4", "primaryDocument": "f4.xml", "items": ""},  # Form 4: ignored
]


@pytest.fixture
def sec(fake_sec, monkeypatch):
    """Fake SEC that counts calls and can 'publish' new filings."""
    state = {"published": False, "companyfacts_calls": 0}

    def fake_get_json(url, ttl=0):
        if url.endswith("company_tickers.json"):
            return load("company_tickers.json")
        if "submissions-001" in url:
            return load("nvda_submissions_001.json")
        if "/submissions/" in url:
            sub = copy.deepcopy(load("nvda_submissions.json"))
            if state["published"]:
                recent = sub["filings"]["recent"]
                for row in reversed(NEW_FILINGS):
                    for k, v in row.items():
                        recent[k].insert(0, v)
            return sub
        if "/companyfacts/" in url:
            state["companyfacts_calls"] += 1
            return load("nvda_companyfacts.json")
        raise AssertionError(url)

    monkeypatch.setattr(sec_client, "get_json", fake_get_json)
    return state


@pytest.fixture
def conn(fake_sec):
    return db.connect(fake_sec / "collect.db")


def run(conn, tickers=("NVDA",)):
    results = collector.collect(conn, list(tickers), today=TODAY)
    for r in results:
        if r["ticker"] == "NVDA" and "error" in r:
            raise AssertionError(f"collector failed for NVDA: {r['error']}")
    return results


def test_first_run_is_a_baseline_without_events(sec, conn):
    (r,) = run(conn)
    assert r["baseline"] and len(r["new_filings"]) > 5 and r["events"] == 0
    assert api.get_pending_events(conn=conn) == []
    assert api.get_company("NVDA", conn=conn)["name"] == "NVIDIA CORP"


def test_nothing_new_skips_the_heavy_load(sec, conn):
    run(conn)
    calls = sec["companyfacts_calls"]
    (r,) = run(conn)
    assert r["new_filings"] == [] and r["events"] == 0
    assert sec["companyfacts_calls"] == calls  # figures not re-downloaded


def test_new_filings_are_stored_and_raise_events(sec, conn):
    run(conn)
    sec["published"] = True
    (r,) = run(conn)
    assert len(r["new_filings"]) == 2 and r["events"] == 2  # the Form 4 is ignored

    stored = {f["accession"] for f in api.list_filings("NVDA", "8-K", limit=50, conn=conn)}
    assert {"0001045810-26-000900", "0001045810-26-000901"} <= stored

    events = {e["accession"]: e for e in api.get_pending_events(conn=conn)}
    earnings = events["0001045810-26-000900"]
    assert earnings["event_type"] == "earnings_8k"
    assert earnings["fiscal_label"] == "Q3 FY2027"          # quarter the earnings 8-K reports on
    assert earnings["document_url"].endswith("nvda-8k-earnings.htm")
    assert events["0001045810-26-000901"]["event_type"] == "other_8k"


def test_events_are_raised_once_and_can_be_marked_done(sec, conn):
    run(conn)
    sec["published"] = True
    run(conn)
    run(conn)                                               # same filings again: no duplicates
    pending = api.get_pending_events(conn=conn)
    assert len(pending) == 2
    assert api.mark_events_processed([e["id"] for e in pending], conn=conn) == 2
    assert api.get_pending_events(conn=conn) == []
    assert api.mark_events_processed([pending[0]["id"]], conn=conn) == 0


def test_one_bad_ticker_does_not_stop_the_run(sec, conn):
    results = {r["ticker"]: r for r in run(conn, ("NVDA", "BADTICKER"))}
    assert "error" in results["BADTICKER"] and "error" not in results["NVDA"]
    row = conn.execute("SELECT status, error FROM ingest_log WHERE ticker='BADTICKER'").fetchone()
    assert row["status"] == "error" and "BADTICKER" in row["error"]


def test_event_type_mapping():
    assert collector.event_type("10-K", []) == "annual_report"
    assert collector.event_type("10-Q", []) == "quarterly_report"
    assert collector.event_type("8-K", ["2.02", "9.01"]) == "earnings_8k"
    assert collector.event_type("8-K", ["5.07"]) == "other_8k"


def test_job_cache_cap_refetches_old_sec_lists(monkeypatch, tmp_path):
    """The 24-hour cache would hide new filings; the job caps cache age while it runs."""
    monkeypatch.setattr(sec_client, "CACHE_DIR", tmp_path)
    fetched = []

    class Resp:
        def json(self):
            return {"n": len(fetched)}

    monkeypatch.setattr(sec_client, "_fetch", lambda url: fetched.append(url) or Resp())
    url = "https://data.sec.gov/submissions/CIK0000000001.json"
    sec_client.get_json(url)
    path = sec_client._cache_path(url, ".json")
    old = time.time() - 600                                 # cached 10 minutes ago
    os.utime(path, (old, old))

    sec_client.get_json(url)
    assert len(fetched) == 1                                # normal: 24 h cache still valid
    monkeypatch.setattr(sec_client, "MAX_JSON_AGE", 300)
    sec_client.get_json(url)
    assert len(fetched) == 2                                # job: older than 5 min -> refetch


def test_collect_restores_max_json_age(sec, conn):
    sec_client.MAX_JSON_AGE = None
    run(conn)
    assert sec_client.MAX_JSON_AGE is None
