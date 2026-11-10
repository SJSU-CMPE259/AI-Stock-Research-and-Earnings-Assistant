"""Download the press release (Exhibit 99.1) of an earnings 8-K.

datastore.ingest saves only an 8-K's main document, which is a one-page cover:
"Item 2.02 ... see Exhibit 99.1". The actual earnings text (results, CEO and CFO
quotes, outlook) is in the exhibit, so the RAG index needs it as well.

The file goes next to the main document: data/filings/<TICKER>/<accession>.ex99.html
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from finsight.rag.index import exhibit_path
from finsight.tools import sec_client

EX99_RE = re.compile(r"ex[-_]?99(?:[-_.]?0?1)?\b|ex99", re.IGNORECASE)


def pick_ex99(names: list[str]) -> str | None:
    """Choose Exhibit 99.1 from a filing folder listing (prefers 99.1 over 99.2)."""
    htm = [n for n in names if n.lower().endswith((".htm", ".html")) and EX99_RE.search(n)]
    if not htm:
        return None
    first = [n for n in htm if re.search(r"99[-_.]?0?1(?!\d)", n)]
    return sorted(first or htm, key=len)[0]


def download_press_release(conn: sqlite3.Connection, accession: str) -> Path | None:
    """Save an earnings 8-K's Exhibit 99.1; None if it isn't an 8-K or has no exhibit."""
    f = conn.execute("SELECT * FROM filings WHERE accession=?", (accession,)).fetchone()
    if not f or not f["form"].startswith("8-K") or not f["local_path"]:
        return None
    dest = exhibit_path(f["local_path"])
    if dest.exists():
        return dest
    folder = f["document_url"].rsplit("/", 1)[0] + "/"
    listing = sec_client.get_json(folder + "index.json")
    names = [i["name"] for i in listing.get("directory", {}).get("item", [])]
    name = pick_ex99(names)
    if not name:
        return None
    dest.write_text(sec_client.get_text(folder + name))
    return dest


def download_all(conn: sqlite3.Connection, tickers: list[str] | None = None) -> list[dict]:
    sql = "SELECT accession, ticker FROM filings WHERE form LIKE '8-K%' AND local_path IS NOT NULL"
    out = []
    for r in conn.execute(sql).fetchall():
        if tickers and r["ticker"] not in {t.upper() for t in tickers}:
            continue
        try:
            p = download_press_release(conn, r["accession"])
            out.append({"accession": r["accession"], "path": str(p) if p else None})
        except Exception as e:  # noqa: BLE001
            out.append({"accession": r["accession"], "error": f"{type(e).__name__}: {e}"})
    return out
