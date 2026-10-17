"""D1 + D3: the loader builds the database, and the shared functions read it correctly."""

from conftest import TODAY

from finsight.datastore import api, ingest


def test_tables_filled(built_db):
    counts = {t: built_db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("companies", "filings", "financial_facts", "ingest_log")}
    assert all(counts.values()), counts


def test_get_company(built_db):
    assert api.get_company("nvda", conn=built_db) == {
        "ticker": "NVDA", "cik": "0001045810", "name": "NVIDIA CORP", "fiscal_year_end": "0125"}


def test_list_filings_reads_older_pages_and_skips_noise(built_db):
    all_f = api.list_filings("NVDA", limit=100, conn=built_db)
    forms = {f["form"] for f in all_f}
    assert forms == {"10-K", "10-Q", "8-K"}               # no Form 4, no 10-Q/A
    assert any(f["filing_date"] < "2025-06-01" for f in all_f)  # came from page 001
    assert all(f["filing_date"] >= "2023-10-16" for f in all_f)  # 3-year window


def test_filing_fiscal_labels(built_db):
    latest_10q = api.list_filings("NVDA", "10-Q", limit=1, conn=built_db)[0]
    assert latest_10q["fiscal_label"] == "Q2 FY2027"
    tenk = api.list_filings("NVDA", "10-K", limit=1, conn=built_db)[0]
    assert tenk["fiscal_label"] == "FY2026"


def test_earnings_8k_items_and_label(built_db):
    eights = {f["filing_date"]: f for f in api.list_filings("NVDA", "8-K", limit=10, conn=built_db)}
    assert eights["2026-08-20"]["items"] == ["2.02", "9.01"]
    assert eights["2026-08-20"]["fiscal_label"] == "Q2 FY2027"
    assert eights["2026-06-25"]["fiscal_label"] is None      # not an earnings 8-K
    assert eights["2026-06-25"]["local_path"] is None        # not downloaded
    assert eights["2026-08-20"]["local_path"]                # earnings 8-K downloaded


def test_get_metric_matches_spec_example(built_db):
    r = api.get_metric("NVDA", "revenue", "quarter", n=1, conn=built_db)[0]
    assert r["fiscal_label"] == "Q2 FY2027"
    assert r["value"] == 96_221_000_000 and r["unit"] == "USD"
    assert r["change_vs_prior_quarter_pct"] == 17.9
    assert r["change_vs_prior_year_pct"] == 105.9            # vs Q2 FY2026
    assert r["derived"] is False and r["form"] == "10-Q"
    assert r["source_url"].endswith("/")


def test_get_metric_derived_q4_and_annual(built_db):
    rows = {r["fiscal_label"]: r for r in api.get_metric("NVDA", "revenue", "quarter", 12, conn=built_db)}
    assert rows["Q4 FY2026"]["derived"] is True
    assert rows["Q1 FY2027"]["change_vs_prior_quarter_pct"] is not None  # Q4 -> Q1 is contiguous
    annual = api.get_metric("NVDA", "revenue", "annual", conn=built_db)
    assert [a["fiscal_label"] for a in annual] == ["FY2026", "FY2025"]
    assert annual[0]["change_vs_prior_quarter_pct"] is None
    assert annual[0]["change_vs_prior_year_pct"] == round((216.0 / 130.497 - 1) * 100, 1)


def test_get_metric_bad_period_type(built_db):
    import pytest
    with pytest.raises(ValueError):
        api.get_metric("NVDA", "revenue", "monthly", conn=built_db)


def test_get_filing_text(built_db):
    acc = api.list_filings("NVDA", "10-Q", limit=1, conn=built_db)[0]["accession"]
    doc = api.get_filing_text(acc, conn=built_db)
    assert "Revenue grew." in doc["text"]
    assert "hidden xbrl" not in doc["text"]
    assert doc["source_url"].startswith("https://www.sec.gov/Archives/")


def test_rerun_is_idempotent_and_reports_no_new_filings(built_db):
    before = built_db.execute("SELECT COUNT(*) FROM financial_facts").fetchone()[0]
    stats = ingest.ingest_company(built_db, "NVDA", today=TODAY, run_started="2026-10-16T00:00:00")
    assert stats["new_filings"] == 0 and stats["new_facts"] == 0
    assert built_db.execute("SELECT COUNT(*) FROM financial_facts").fetchone()[0] == before
    assert api.check_new_filings(conn=built_db) == []


def test_check_new_filings_finds_added(built_db):
    built_db.execute("DELETE FROM filings WHERE filing_date='2026-08-20'")  # pretend it's new
    ingest.ingest_company(built_db, "NVDA", today=TODAY, run_started="2026-10-16T00:00:00")
    new = api.check_new_filings(conn=built_db)
    assert [f["filing_date"] for f in new] == ["2026-08-20"]


def test_quality_checks_pass(built_db):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "check_quality", Path(__file__).resolve().parents[1] / "scripts" / "check_quality.py")
    cq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cq)
    results = cq.check(built_db, today=TODAY)
    assert not [r for r in results if r[0] == "FAIL"], results
