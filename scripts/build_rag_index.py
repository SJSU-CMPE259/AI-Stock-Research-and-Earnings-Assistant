"""Build the RAG vector index from the filings already downloaded by scripts/build_db.py.

    python scripts/build_rag_index.py                      # index everything new
    python scripts/build_rag_index.py --tickers AAPL NVDA  # some companies
    python scripts/build_rag_index.py --press-releases     # also fetch 8-K Exhibit 99.1 first
    python scripts/build_rag_index.py --force              # re-embed every filing
    python scripts/build_rag_index.py --rebuild            # drop the collection first (new model)
    python scripts/build_rag_index.py --ask "What does Apple say about tariffs?" --ticker AAPL
    python scripts/build_rag_index.py --ask "..." --ticker AAPL --mode sparse   # or dense / hybrid

Vector DB location: FINSIGHT_QDRANT_URL (server) or data/qdrant/ (embedded). Embedding
model: FINSIGHT_EMBED_MODEL (default BAAI/bge-small-en-v1.5). See src/finsight/rag/README.md.
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) # run from anywhere

from finsight.datastore import db
from finsight.rag import SEARCH_MODES, exhibits, get_store, guess_topics, index_all, retrieve, search
from finsight.rag.retrieve import DEFAULT_BUDGET, latest_accessions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--press-releases", action="store_true")
    ap.add_argument("--ask")
    ap.add_argument("--ticker")
    ap.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")
    ap.add_argument("--filings", type=int, default=1, help="how many recent filings to use")
    ap.add_argument("--forms", nargs="*", default=["10-K", "10-Q"], help="filing types to answer from")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="evidence token budget")
    ap.add_argument("--search", action="store_true",
                    help="ranked search only (skip whole-section mode): use it to compare --mode")
    ap.add_argument("-k", type=int, default=8, help="results to show with --search")
    a = ap.parse_args()
    conn, store = db.connect(), get_store()
    print(f"vector DB: {store.collection} · embedder: {store.dense.name} ({store.dense.dim}-dim)")

    if a.ask:
        if not a.ticker:
            ap.error("--ask needs --ticker")
        if a.search:                       # pure ranked search: what --mode actually changes
            topics = guess_topics(a.ask) or None
            accs = latest_accessions(a.ticker, tuple(a.forms), a.filings, topics[0] if topics else None,
                                     conn=conn, store=store)
            hits = search(a.ask, a.ticker, topics, accs or None, k=a.k, mode=a.mode, store=store)
            print(f"ranked search · mode={a.mode} · topics={topics} · filings={accs} · top {len(hits)}\n")
            for i, e in enumerate(hits, 1):
                print(f"#{i} score={e.score:.3f}  {e.passage_id.split('/', 1)[1]}  {e.form} {e.fiscal_label} · "
                      f"{e.section[:50]}\n     {e.text[:240]}...\n")
            return
        r = retrieve(a.ask, a.ticker, forms=tuple(a.forms), n_filings=a.filings,
                     budget_tokens=a.budget, search_mode=a.mode, conn=conn, store=store)
        accs = sorted({e.passage_id.split("/")[0] for e in r["evidence"]})
        print(f"mode={r['mode']} search={r['search_mode']} topics={r['topics']}  "
              f"~{r['tokens']:,} of {a.budget:,} tokens  {len(r['evidence'])} passages from {accs}")
        if r["mode"] == "section":
            print("  -> the whole section fit the budget, so it was returned in reading order and no\n"
                  "     search ran: --mode has no effect here. Compare modes with --search, or a\n"
                  "     smaller --budget.")
        print()
        for i, e in enumerate(r["evidence"][:10], 1):
            sc = f" score={e.score:.3f}" if e.score is not None else ""
            print(f"[S{i}]{sc} {e.passage_id.split('/', 1)[1]}  {e.citation()}\n     {e.text[:220]}...\n")
        if len(r["evidence"]) > 10:
            print(f"... {len(r['evidence']) - 10} more")
        return

    if a.rebuild:
        store.drop()
        print("dropped collection; re-indexing everything")
    if a.press_releases:
        got = exhibits.download_all(conn, a.tickers)
        print(f"press releases: {sum(1 for g in got if g.get('path'))} saved, "
              f"{sum(1 for g in got if 'error' in g)} errors")

    t0 = time.time()

    def progress(results, done, total):
        n = sum(r.get("passages", 0) for r in results if "error" not in r and not r.get("skipped"))
        el = time.time() - t0
        rate = f" · {n / el:,.0f} paragraphs/s" if n and el > 0 else ""
        print(f"  {done}/{total} filings · {n:,} paragraphs embedded · {el:,.0f}s{rate}", flush=True)

    results = index_all(conn, a.tickers, force=a.force or a.rebuild, store=store, progress=progress)
    done = [r for r in results if "passages" in r]
    print(f"indexed {len(done)} filings ({sum(r['passages'] for r in done):,} paragraphs) in "
          f"{time.time() - t0:.0f}s · skipped {sum(1 for r in results if r.get('skipped'))} · "
          f"errors {sum(1 for r in results if 'error' in r)} · total in DB {store.count():,}")
    for r in done:
        flag = "" if r["found_items"] else "  <- no Item headings found, indexed as full text"
        print(f"  {r['accession']} {r['form']:5} {r['sections']:3} sections "
              f"{r['passages']:5} paragraphs{flag}")
    for r in results:
        if "error" in r:
            print(f"  ERROR {r['accession']}: {r['error']}")


if __name__ == "__main__":
    main()
