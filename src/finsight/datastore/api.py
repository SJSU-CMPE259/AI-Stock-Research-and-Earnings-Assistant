"""The functions other workstreams call. They read from OUR database, never the live SEC.

    get_company(ticker)
    list_filings(ticker, form, limit)
    get_filing_text(accession)
    get_metric(ticker, metric, period_type, n)
    check_new_filings(since)

Every function takes an optional `conn`; without it, the default database
(config.DB_PATH, built by scripts/build_db.py) is used.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from finsight.datastore import db
from finsight.tools.xbrl import fiscal_label

PERIOD_TYPES = {"quarter": "quarter", "quarterly": "quarter", "q": "quarter",
                "annual": "annual", "year": "annual", "fy": "annual",
                "instant": "instant", "balance": "instant"}


def _conn(conn: sqlite3.Connection | None) -> sqlite3.Connection:
    return conn or db.connect()


def get_company(ticker: str, conn=None) -> dict | None:
    row = _conn(conn).execute(
        "SELECT ticker, cik, name, fiscal_year_end FROM companies WHERE ticker=?",
        (ticker.upper(),),
    ).fetchone()
    return dict(row) if row else None


def list_filings(ticker: str, form: str | None = None, limit: int = 10, conn=None) -> list[dict]:
    """Newest first, with fiscal labels and links."""
    sql = "SELECT * FROM filings WHERE ticker=?"
    args: list = [ticker.upper()]
    if form:
        sql += " AND form=?"
        args.append(form)
    sql += " ORDER BY filing_date DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in _conn(conn).execute(sql, args):
        d = dict(r)
        d["items"] = [x for x in (d["items"] or "").split(",") if x]
        d["fiscal_label"] = (fiscal_label(d["fiscal_year"], d["fiscal_period"])
                             if d["fiscal_year"] else None)
        out.append(d)
    return out


class _TextExtractor(HTMLParser):
    """Very simple HTML -> text. Skips scripts, styles and the hidden inline-XBRL header.

    Good enough for a first look; proper section splitting (Item 1A, Item 7...)
    is the retrieval workstream's job.
    """
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
    SKIP = {"script", "style", "ix:header", "head"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def get_filing_text(accession: str, conn=None) -> dict | None:
    """Cleaned plain text of a downloaded filing, plus its source link."""
    row = _conn(conn).execute("SELECT * FROM filings WHERE accession=?", (accession,)).fetchone()
    if not row:
        return None
    path = row["local_path"]
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{accession} not downloaded; run scripts/build_db.py")
    return {
        "accession": accession, "ticker": row["ticker"], "form": row["form"],
        "filing_date": row["filing_date"], "source_url": row["document_url"],
        "text": html_to_text(Path(path).read_text(errors="replace")),
    }


def _pct(new: float, old: float | None) -> float | None:
    if old is None or old <= 0:  # % change from zero/negative isn't meaningful
        return None
    return round((new / old - 1) * 100, 1)


def _contiguous(prev_end: str, start: str | None) -> bool:
    """True if `start` follows right after `prev_end` (no missing quarter in between)."""
    if not start:
        return True
    gap = (date.fromisoformat(start) - date.fromisoformat(prev_end)).days
    return 0 < gap <= 7


def get_metric(ticker: str, metric: str, period_type: str = "quarter", n: int = 8,
               conn=None) -> list[dict]:
    """Most recent n periods, newest first, with fiscal label, value, unit, changes and source.

    period_type: "quarter", "annual", or "instant" (balance-sheet snapshot).
    """
    ptype = PERIOD_TYPES.get(period_type.lower())
    if not ptype:
        raise ValueError(f"period_type must be one of {sorted(set(PERIOD_TYPES.values()))}")
    rows = [dict(r) for r in _conn(conn).execute(
        """SELECT * FROM financial_facts WHERE ticker=? AND metric=? AND period_type=?
           ORDER BY period_end""",
        (ticker.upper(), metric, ptype),
    )]
    by_fiscal = {(r["fiscal_year"], r["fiscal_period"]): r for r in rows}

    out = []
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        prior_q = None
        if ptype != "annual" and prev and (ptype == "instant" or
                                           _contiguous(prev["period_end"], r["period_start"])):
            prior_q = prev["value"]
        last_year = by_fiscal.get((r["fiscal_year"] - 1, r["fiscal_period"]))
        out.append({
            "ticker": r["ticker"], "metric": r["metric"],
            "fiscal_year": r["fiscal_year"], "fiscal_period": r["fiscal_period"],
            "fiscal_label": fiscal_label(r["fiscal_year"], r["fiscal_period"]),
            "period_start": r["period_start"], "period_end": r["period_end"],
            "value": r["value"], "unit": r["unit"],
            "change_vs_prior_quarter_pct": _pct(r["value"], prior_q),
            "change_vs_prior_year_pct": _pct(r["value"], last_year["value"] if last_year else None),
            "derived": bool(r["derived"]), "form": r["form"], "filed": r["filed"],
            "accession": r["accession"], "concept": r["concept"], "source_url": r["source_url"],
        })
    return list(reversed(out))[:n]


def check_new_filings(since: str | None = None, conn=None) -> list[dict]:
    """Filings stored since `since` (ISO time). Default: since the previous loader run started.

    The automatic collection job (D4) runs the loader, then calls this to raise
    "new filing" events.
    """
    c = _conn(conn)
    if since is None:
        runs = c.execute(
            "SELECT DISTINCT started_at FROM ingest_log WHERE status='ok' ORDER BY started_at DESC LIMIT 2"
        ).fetchall()
        if len(runs) < 2:
            return []  # first run: everything is "new", so report nothing
        since = runs[0]["started_at"]
    rows = c.execute(
        "SELECT * FROM filings WHERE added_at >= ? ORDER BY filing_date DESC", (since,)
    ).fetchall()
    return [dict(r) for r in rows]
