"""Store earnings-call transcripts: which quarters to fetch, fetch them, keep a record.

    ingest_transcripts(conn, ["NVDA", "AAPL"], quarters=8)

Each transcript is saved as JSON in data/transcripts/<TICKER>/<YYYYQn>.json and gets one
row in the `transcripts` table (created here, so schema.sql doesn't change). The RAG index
(finsight.rag.index.index_transcripts) reads both.

Quarters come from OUR filings table: the fiscal labels of earnings 8-Ks, 10-Qs and 10-Ks
(a 10-K's "FY" is the Q4 call). So transcripts line up with the filings they discuss.
`call_date` is the earnings 8-K's filing date (the call is usually that day), or the
10-Q/10-K date if there is no 8-K.

Safe to run again: quarters already stored are skipped. A quarter with no transcript yet
("empty") is asked again after a day. The free key allows 25 calls a day; when the limit is
hit the run stops cleanly and the next run continues where it left off.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from finsight import config
from finsight.tools import transcripts as tx

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transcripts (
    doc_id          TEXT PRIMARY KEY,          -- "TR-NVDA-2027Q2": the RAG index uses it like an accession
    ticker          TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    fiscal_period   TEXT NOT NULL,             -- Q1..Q4
    quarter         TEXT NOT NULL,             -- "2027Q2", as sent to the provider
    call_date       TEXT,                      -- approx: earnings 8-K filing date
    status          TEXT NOT NULL,             -- ok / empty
    n_turns         INTEGER DEFAULT 0,
    n_words         INTEGER DEFAULT 0,
    local_path      TEXT,
    source          TEXT,
    source_url      TEXT,                      -- no API key in it
    fetched_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_ticker ON transcripts(ticker, fiscal_year, fiscal_period);
"""
RETRY_EMPTY_AFTER = timedelta(days=1)


def transcripts_dir() -> Path:
    return Path(config.DB_PATH).parent / "transcripts"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(TABLE_SQL)


def doc_id(ticker: str, quarter: str) -> str:
    return f"TR-{ticker.upper()}-{quarter}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quarters_to_fetch(conn: sqlite3.Connection, ticker: str, n: int = 8) -> list[dict]:
    """Newest n fiscal quarters we have filings for: [{fiscal_year, fiscal_period, quarter, call_date}]."""
    rows = conn.execute(
        """SELECT form, items, filing_date, fiscal_year, fiscal_period FROM filings
           WHERE ticker=? AND fiscal_year IS NOT NULL AND fiscal_period IS NOT NULL
             AND (form IN ('10-Q', '10-K') OR (form = '8-K' AND items LIKE '%2.02%'))""",
        (ticker.upper(),)).fetchall()
    by_q: dict[tuple[int, str], dict] = {}
    for form, _items, filed, fy, fp in rows:
        fp = "Q4" if fp == "FY" else fp
        if fp not in ("Q1", "Q2", "Q3", "Q4"):
            continue
        d = by_q.setdefault((int(fy), fp), {"fiscal_year": int(fy), "fiscal_period": fp,
                                            "quarter": tx.quarter_key(fy, fp), "call_date": None,
                                            "_report": None})
        if form == "8-K":       # earnings release: the call is usually the same day
            d["call_date"] = min(filter(None, [d["call_date"], filed]))
        else:
            d["_report"] = min(filter(None, [d["_report"], filed]))
    out = sorted(by_q.values(), key=lambda d: (d["fiscal_year"], d["fiscal_period"]), reverse=True)[:n]
    for d in out:
        d["call_date"] = d["call_date"] or d.pop("_report")
        d.pop("_report", None)
    return out


def _needs_fetch(conn, did: str, force: bool) -> bool:
    row = conn.execute("SELECT status, fetched_at FROM transcripts WHERE doc_id=?", (did,)).fetchone()
    if force or row is None:
        return True
    if row[0] == "ok":
        return False
    return datetime.fromisoformat(row[1]) < datetime.now(timezone.utc) - RETRY_EMPTY_AFTER


def save_transcript(conn: sqlite3.Connection, t: tx.Transcript, q: dict) -> dict:
    """Write the JSON file and the table row for one fetched quarter."""
    did = doc_id(t.ticker, q["quarter"])
    path = None
    if t.available:
        path = transcripts_dir() / t.ticker / f"{q['quarter']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(t.to_dict(), indent=1))
    conn.execute(
        """INSERT INTO transcripts (doc_id, ticker, fiscal_year, fiscal_period, quarter, call_date,
             status, n_turns, n_words, local_path, source, source_url, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(doc_id) DO UPDATE SET call_date=excluded.call_date, status=excluded.status,
             n_turns=excluded.n_turns, n_words=excluded.n_words, local_path=excluded.local_path,
             source=excluded.source, source_url=excluded.source_url, fetched_at=excluded.fetched_at""",
        (did, t.ticker, q["fiscal_year"], q["fiscal_period"], q["quarter"], q.get("call_date"),
         "ok" if t.available else "empty", len(t.turns), t.n_words,
         str(path) if path else None, t.source, t.source_url, _now()))
    conn.commit()
    return {"doc_id": did, "ticker": t.ticker, "quarter": q["quarter"],
            "status": "ok" if t.available else "empty", "turns": len(t.turns), "words": t.n_words}


def ingest_transcripts(conn: sqlite3.Connection, tickers: list[str], quarters: int = 8,
                       max_calls: int = 25, force: bool = False, fetch=None) -> dict:
    """Fetch and store missing transcripts. Returns {"results": [...], "calls": n, "stopped": reason|None}.

    fetch(ticker, quarter) -> Transcript; defaults to the Alpha Vantage tool (tests pass a fake).
    """
    ensure_table(conn)
    fetch = fetch or tx.get_transcript
    results, calls, stopped = [], 0, None
    for ticker in tickers:
        for q in quarters_to_fetch(conn, ticker, quarters):
            did = doc_id(ticker, q["quarter"])
            if not _needs_fetch(conn, did, force):
                results.append({"doc_id": did, "ticker": ticker.upper(), "quarter": q["quarter"],
                                "status": "skipped"})
                continue
            if calls >= max_calls:
                return {"results": results, "calls": calls, "stopped": f"max_calls={max_calls} reached"}
            try:
                calls += 1
                results.append(save_transcript(conn, fetch(ticker, q["quarter"]), q))
            except tx.TranscriptLimitError as e:
                return {"results": results, "calls": calls, "stopped": str(e)}
            except tx.TranscriptAccessError:
                raise                    # bad key / premium: every call would fail the same way
            except Exception as e:  # noqa: BLE001 — one bad quarter shouldn't stop the rest
                results.append({"doc_id": did, "ticker": ticker.upper(), "quarter": q["quarter"],
                                "status": "error", "error": f"{type(e).__name__}: {e}"})
    return {"results": results, "calls": calls, "stopped": stopped}


def list_transcripts(conn: sqlite3.Connection, ticker: str | None = None,
                     status: str = "ok") -> list[dict]:
    try:
        sql, args = "SELECT * FROM transcripts WHERE status=?", [status]
        if ticker:
            sql += " AND ticker=?"
            args.append(ticker.upper())
        cur = conn.execute(sql + " ORDER BY call_date DESC", args)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except sqlite3.OperationalError:     # table not created yet: no transcripts ingested
        return []
