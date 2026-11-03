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

import os
import sqlite3
from pathlib import Path
from typing import Callable

from finsight.datastore.api import html_to_text
from finsight.rag.sections import SPLITTER_VERSION, Section, split_exhibit, split_filing
from finsight.rag.store import EMBED_CHUNK, VectorStore, get_store
from finsight.tools.xbrl import fiscal_label


# Sections to leave out of the index, e.g. "exhibits,summary" (comma-separated topic names).
# Saves embedding time; re-run with --force after changing it.
SKIP_TOPICS = {t.strip() for t in os.getenv("FINSIGHT_RAG_SKIP_TOPICS", "").split(",") if t.strip()}


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


def _prepare(conn: sqlite3.Connection, accession: str, force: bool, store: VectorStore) -> dict:
    """Read and split one filing (no embedding yet). Returns its result row + payloads."""
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
    payloads = [p for p in _payloads(f, sections) if p["topic"] not in SKIP_TOPICS]
    return {"accession": accession, "ticker": f["ticker"], "form": f["form"],
            "sections": len(sections), "passages": len(payloads), "found_items": found_items,
            "topics": sorted({s.topic for s in sections}), "_payloads": payloads}


def _store(store: VectorStore, prepared: list[dict]) -> None:
    for r in prepared:
        store.delete_filing(r["accession"])   # drop paragraphs a new splitter no longer makes
    store.upsert([p for r in prepared for p in r.pop("_payloads")])


def index_filing(conn: sqlite3.Connection, accession: str, force: bool = False,
                 store: VectorStore | None = None) -> dict:
    """Split one downloaded filing, embed its paragraphs and store them in the vector DB."""
    store = store or get_store()
    r = _prepare(conn, accession, force, store)
    if not r.get("skipped"):
        _store(store, [r])
    return r


def index_all(conn: sqlite3.Connection, tickers: list[str] | None = None,
              forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"), force: bool = False,
              store: VectorStore | None = None,
              progress: Callable[[list[dict], int, int], None] | None = None) -> list[dict]:
    """Index every downloaded filing. One bad filing doesn't stop the others.

    Paragraphs from several filings are embedded together (EMBED_CHUNK at a time): the
    model is much faster on big batches than on one 10-Q at a time.
    progress(results_so_far, filings_done, filings_total) is called after each batch.
    """
    store = store or get_store()
    sql = (f"SELECT accession FROM filings WHERE local_path IS NOT NULL "
           f"AND form IN ({','.join('?' * len(forms))})")
    args: list = list(forms)
    if tickers:
        sql += f" AND ticker IN ({','.join('?' * len(tickers))})"
        args += [t.upper() for t in tickers]
    accs = [a for (a,) in conn.execute(sql + " ORDER BY filing_date DESC", args).fetchall()]
    results: list[dict] = []
    buffer: list[dict] = []
    last_reported = 0

    def flush():
        if not buffer:
            return
        try:
            _store(store, buffer)
        except Exception as e:  # noqa: BLE001 — mark the whole batch as failed, keep going
            for r in buffer:
                r.pop("_payloads", None)
                r["error"] = f"{type(e).__name__}: {e}"
        buffer.clear()

    for n, acc in enumerate(accs, 1):
        try:
            r = _prepare(conn, acc, force, store)
        except Exception as e:  # noqa: BLE001 — log and keep going
            r = {"accession": acc, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "_payloads" in r:
            buffer.append(r)
            if sum(len(b["_payloads"]) for b in buffer) >= EMBED_CHUNK:
                flush()
                if progress:
                    progress(results, n, len(accs))
                    last_reported = n
    had_tail = bool(buffer)
    flush()
    if progress and (had_tail or last_reported != len(accs)):
        progress(results, len(accs), len(accs))
    return results
