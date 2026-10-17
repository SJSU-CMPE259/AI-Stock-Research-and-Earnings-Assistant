"""Loader: fetch from the SEC, clean, and store in SQLite.

Used by scripts/build_db.py now, and by the automatic collection job (D4) later.
Safe to run again: existing rows are updated, nothing is duplicated.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from finsight import config
from finsight.tools import edgar, xbrl

NEXT_PERIOD = {"Q1": "Q2", "Q2": "Q3", "Q3": "Q4", "Q4": "Q1"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _quarter_for_8k(filing_date: str, periods: list[tuple[str, int, str]]) -> tuple[int, str] | None:
    """Fiscal quarter an earnings 8-K reports on: the latest quarter ended before it was filed.

    `periods` = sorted (period_end, fiscal_year, fiscal_period) from 10-Qs/10-Ks.
    If the 10-Q isn't out yet (8-Ks usually come first), step one quarter forward.
    """
    before = [p for p in periods if p[0] < filing_date]
    if not before:
        return None
    end, fy, fp = before[-1]
    if (date.fromisoformat(filing_date) - date.fromisoformat(end)).days > 75:
        fp = NEXT_PERIOD[fp]
        fy = fy + 1 if fp == "Q1" else fy
    return fy, fp


def ingest_company(conn: sqlite3.Connection, ticker: str, years: int = 3,
                   download_docs: bool = True, today: date | None = None,
                   overrides: dict | None = None, run_started: str | None = None) -> dict:
    """Load one company's filings and figures. Returns counts for the log.

    run_started: one timestamp shared by every company in the same loader run,
    so check_new_filings() can find what this run added.
    """
    today = today or date.today()
    since = (today - timedelta(days=365 * years)).isoformat()
    started = run_started or _now()
    stats = {"ticker": ticker.upper(), "new_filings": 0, "new_facts": 0}

    # 1. Company details
    info = edgar.get_company_info(ticker)
    conn.execute(
        """INSERT INTO companies (ticker, cik, name, fiscal_year_end, updated_at)
           VALUES (:ticker, :cik, :name, :fiscal_year_end, :updated_at)
           ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik, name=excluded.name,
             fiscal_year_end=excluded.fiscal_year_end, updated_at=excluded.updated_at""",
        {**info, "updated_at": started},
    )

    # 2. Figures (also gives us the fiscal label of every 10-K / 10-Q)
    facts_json = xbrl.get_company_facts(ticker)
    fiscal = xbrl.filing_periods(facts_json)
    quarter_ends = sorted(
        (end, fy, "Q4" if fp == "FY" else fp) for fy, fp, end in fiscal.values()
    )

    # 3. Filings list (+ documents)
    for f in edgar.list_filings_since(ticker, since):
        fy, fp = fiscal.get(f.accession, (None, None))[:2]
        if f.form == "8-K" and f.is_earnings_8k:
            fy, fp = _quarter_for_8k(f.filing_date, quarter_ends) or (None, None)

        local_path = None
        if download_docs and (f.form in ("10-K", "10-Q") or f.is_earnings_8k):
            local_path = str(edgar.download_document(f, config.FILINGS_DIR))

        exists = conn.execute("SELECT 1 FROM filings WHERE accession=?", (f.accession,)).fetchone()
        conn.execute(
            """INSERT INTO filings (accession, ticker, form, filing_date, period_end, fiscal_year,
                 fiscal_period, items, document_url, local_path, added_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(accession) DO UPDATE SET fiscal_year=excluded.fiscal_year,
                 fiscal_period=excluded.fiscal_period, items=excluded.items,
                 local_path=COALESCE(excluded.local_path, filings.local_path)""",
            (f.accession, f.ticker, f.form, f.filing_date, f.period_of_report, fy, fp,
             ",".join(f.items), f.url, local_path, started),
        )
        stats["new_filings"] += 0 if exists else 1

    # 4. Figures -> database (keep one extra year so year-over-year changes work)
    keep_from = (today - timedelta(days=365 * (years + 1))).isoformat()
    count_sql = "SELECT COUNT(*) FROM financial_facts WHERE ticker=?"
    before = conn.execute(count_sql, (ticker.upper(),)).fetchone()[0]
    for fact in xbrl.extract_all(facts_json, ticker.upper(), overrides):
        if fact.end < keep_from:
            continue
        conn.execute(
            """INSERT INTO financial_facts (ticker, metric, concept, unit, period_type, period_start,
                 period_end, fiscal_year, fiscal_period, value, form, accession, filed, derived,
                 source_url)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, metric, period_type, period_end) DO UPDATE SET
                 concept=excluded.concept, unit=excluded.unit, period_start=excluded.period_start,
                 fiscal_year=excluded.fiscal_year, fiscal_period=excluded.fiscal_period,
                 value=excluded.value, form=excluded.form, accession=excluded.accession,
                 filed=excluded.filed, derived=excluded.derived, source_url=excluded.source_url""",
            (fact.ticker, fact.metric, fact.concept, fact.unit, fact.period_type,
             fact.start, fact.end, fact.fiscal_year, fact.fiscal_period,
             fact.value, fact.form, fact.accession, fact.filed, int(fact.derived),
             fact.source_url),
        )
    stats["new_facts"] = conn.execute(count_sql, (ticker.upper(),)).fetchone()[0] - before

    conn.execute(
        """INSERT INTO ingest_log (started_at, finished_at, ticker, new_filings, new_facts, status)
           VALUES (?,?,?,?,?, 'ok')""",
        (started, _now(), stats["ticker"], stats["new_filings"], stats["new_facts"]),
    )
    conn.commit()
    return stats


def ingest_all(conn: sqlite3.Connection, tickers: list[str], **kwargs) -> list[dict]:
    """Load every company; one company failing doesn't stop the others."""
    overrides = config.load_concept_overrides()
    started = _now()
    results = []
    for t in tickers:
        try:
            results.append(ingest_company(conn, t, overrides=overrides.get(t.upper()),
                                          run_started=started, **kwargs))
        except Exception as e:  # noqa: BLE001 — log and keep going
            conn.rollback()
            conn.execute(
                "INSERT INTO ingest_log (started_at, finished_at, ticker, status, error) "
                "VALUES (?,?,?, 'error', ?)",
                (started, _now(), t.upper(), f"{type(e).__name__}: {e}"),
            )
            conn.commit()
            results.append({"ticker": t.upper(), "error": f"{type(e).__name__}: {e}"})
    return results


def reset_database(path: Path) -> None:
    """Delete the database file so the loader rebuilds it from scratch."""
    Path(path).unlink(missing_ok=True)
