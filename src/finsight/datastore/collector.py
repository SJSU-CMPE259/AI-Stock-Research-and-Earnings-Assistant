"""Automatic collection job (D4): detect new SEC filings, fetch and store them, raise events.

Each run, for every company:
  1. Cheap check: one SEC call for the company's filing list, compared with our
     database. Most runs find nothing new and stop here.
  2. Only if something is new: run the normal loader (ingest_company), which
     stores the filing, downloads its document and refreshes the figures.
  3. Record a "new filing" event per new 10-K / 10-Q / 8-K in `filing_events`.
     The orchestrator reads pending events (api.get_pending_events), runs
     analysis / alerts, then marks them processed.

The first run for a company is a "baseline": it loads the history but raises no
events, otherwise three years of filings would all look new.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

from finsight import config
from finsight.datastore import ingest
from finsight.tools import edgar, sec_client

EVENT_TYPES = {"10-K": "annual_report", "10-Q": "quarterly_report"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def event_type(form: str, items: list[str]) -> str:
    if form == "8-K":
        return "earnings_8k" if "2.02" in items else "other_8k"
    return EVENT_TYPES.get(form, form)


def find_new_filings(conn: sqlite3.Connection, ticker: str, since: str) -> list[edgar.Filing]:
    """Filings the SEC lists for this company that aren't in our database yet."""
    known = {r[0] for r in conn.execute("SELECT accession FROM filings WHERE ticker=?",
                                        (ticker.upper(),))}
    return [f for f in edgar.list_filings_since(ticker, since) if f.accession not in known]


def collect_company(conn: sqlite3.Connection, ticker: str, years: int = 3,
                    download_docs: bool = True, today: date | None = None,
                    overrides: dict | None = None, run_started: str | None = None) -> dict:
    """Check one company; load and raise events only if it has new filings."""
    today = today or date.today()
    started = run_started or _now()
    t = ticker.upper()
    since = (today - timedelta(days=365 * years)).isoformat()

    baseline = conn.execute("SELECT 1 FROM filings WHERE ticker=? LIMIT 1", (t,)).fetchone() is None
    new = find_new_filings(conn, t, since)
    result = {"ticker": t, "new_filings": [], "events": 0, "baseline": baseline}
    if not new:
        return result

    ingest.ingest_company(conn, t, years=years, download_docs=download_docs, today=today,
                          overrides=overrides, run_started=started)
    result["new_filings"] = [f"{f.form} {f.filing_date} {f.accession}" for f in new]
    if baseline:
        return result  # first load: history, not news

    for f in new:
        row = conn.execute("SELECT fiscal_year, fiscal_period FROM filings WHERE accession=?",
                           (f.accession,)).fetchone()
        cur = conn.execute(
            """INSERT OR IGNORE INTO filing_events (accession, ticker, event_type, form,
                 filing_date, fiscal_year, fiscal_period, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (f.accession, t, event_type(f.form, f.items), f.form, f.filing_date,
             row["fiscal_year"] if row else None, row["fiscal_period"] if row else None, started),
        )
        result["events"] += cur.rowcount
    conn.commit()
    return result


def collect(conn: sqlite3.Connection, tickers: list[str], max_age: int = 300, **kwargs) -> list[dict]:
    """One collection run over all companies. One company failing doesn't stop the others.

    max_age: SEC responses older than this many seconds are re-fetched, so new
    filings show up (the normal cache keeps filing lists for 24 hours).
    """
    overrides = config.load_concept_overrides()
    started = _now()
    previous, sec_client.MAX_JSON_AGE = sec_client.MAX_JSON_AGE, max_age
    results = []
    try:
        for t in tickers:
            try:
                results.append(collect_company(conn, t, overrides=overrides.get(t.upper()),
                                               run_started=started, **kwargs))
            except Exception as e:  # noqa: BLE001 — log it and move on to the next company
                conn.rollback()
                msg = f"{type(e).__name__}: {e}"
                conn.execute("INSERT INTO ingest_log (started_at, finished_at, ticker, status, error) "
                             "VALUES (?,?,?, 'error', ?)", (started, _now(), t.upper(), msg))
                conn.commit()
                results.append({"ticker": t.upper(), "error": msg})
    finally:
        sec_client.MAX_JSON_AGE = previous
    return results
