"""Build the RAG index: downloaded filings -> sections -> paragraphs -> vector DB.

    index_filing(conn, accession)     one filing
    index_all(conn, tickers=None)     every downloaded filing not yet indexed

`conn` is the FinSight SQLite datastore: it is only READ here, for the list of
filings and their local copies. Paragraphs, vectors and metadata live in the
vector DB (see store.py).

Safe to run again: a filing is skipped if the vector DB already has it with the
current SPLITTER_VERSION, unless force=True. Runs after datastore.ingest
(scripts/build_rag_index.py).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from finsight.datastore.api import html_to_text
from finsight.rag.sections import SPLITTER_VERSION, Section, split_exhibit, split_filing
from finsight.rag.store import VectorStore, get_store
from finsight.tools.xbrl import fiscal_label


def exhibit_path(local_path: str | Path) -> Path:
    """Where exhibits.download_press_release saves an 8-K's Exhibit 99.1."""
    p = Path(local_path)
    return p.with_name(p.stem + ".ex99.html")


def _payloads(f: sqlite3.Row, sections: list[Section]) -> list[dict]:
    """One payload per TEXT paragraph. Table rows are left out: numbers come from XBRL."""
    label = fiscal_label(f["fiscal_year"], f["fiscal_period"]) if f["fiscal_year"] else None
    out = []
    for si, sec in enumerate(sections):
        key = f"{sec.part}-{sec.item}" if sec.part else sec.item
        where = (f"Part {sec.part} Item {sec.item}" if sec.part else
                 "Exhibit 99.1" if sec.item.startswith("EX-") else f"Item {sec.item}")
        for p in sec.passages:
            if p.kind != "text":
                continue
            out.append({
                "passage_id": f"{f['accession']}/{key}/p{p.seq:03d}",
                "accession": f["accession"], "ticker": f["ticker"], "form": f["form"],
                "filing_date": f["filing_date"], "fiscal_label": label,
                "document_url": f["document_url"] or "",
                "part": sec.part, "item": sec.item, "topic": sec.topic, "title": sec.title,
                "section": f"{where} · {sec.title}", "section_idx": si, "seq": p.seq,
                "heading": p.heading, "text": p.text, "n_words": p.n_words,
                "splitter_version": SPLITTER_VERSION,
            })
    return out


def index_filing(conn: sqlite3.Connection, accession: str, force: bool = False,
                 store: VectorStore | None = None) -> dict:
    """Split one downloaded filing, embed its paragraphs and store them in the vector DB."""
    store = store or get_store()
    f = conn.execute("SELECT * FROM filings WHERE accession=?", (accession,)).fetchone()
    if not f:
        raise ValueError(f"Unknown filing {accession}")
    if not f["local_path"] or not Path(f["local_path"]).exists():
        raise FileNotFoundError(f"{accession} not downloaded; run scripts/build_db.py")
    if not force and store.count(accessions=[accession], splitter_version=SPLITTER_VERSION):
        return {"accession": accession, "skipped": True}

    text = html_to_text(Path(f["local_path"]).read_text(errors="replace"))
    sections = split_filing(text, f["form"])
    found_items = not (len(sections) == 1 and sections[0].topic == "full_text")
    ex = exhibit_path(f["local_path"])
    if f["form"].startswith("8-K") and ex.exists():
        sections.append(split_exhibit(html_to_text(ex.read_text(errors="replace"))))

    payloads = _payloads(f, sections)
    store.delete_filing(accession)          # drop paragraphs a new splitter no longer makes
    store.upsert(payloads)
    return {"accession": accession, "form": f["form"], "sections": len(sections),
            "passages": len(payloads), "found_items": found_items,
            "topics": sorted({s.topic for s in sections})}


def index_all(conn: sqlite3.Connection, tickers: list[str] | None = None,
              forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"), force: bool = False,
              store: VectorStore | None = None) -> list[dict]:
    """Index every downloaded filing. One bad filing doesn't stop the others."""
    store = store or get_store()
    sql = (f"SELECT accession FROM filings WHERE local_path IS NOT NULL "
           f"AND form IN ({','.join('?' * len(forms))})")
    args: list = list(forms)
    if tickers:
        sql += f" AND ticker IN ({','.join('?' * len(tickers))})"
        args += [t.upper() for t in tickers]
    results = []
    for (acc,) in conn.execute(sql + " ORDER BY filing_date DESC", args).fetchall():
        try:
            results.append(index_filing(conn, acc, force=force, store=store))
        except Exception as e:  # noqa: BLE001 — log and keep going
            results.append({"accession": acc, "error": f"{type(e).__name__}: {e}"})
    return results
