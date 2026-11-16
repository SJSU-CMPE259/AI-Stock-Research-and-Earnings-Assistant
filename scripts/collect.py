"""Collection job (D4): check the SEC for new filings, store them, raise "new filing" events.

Run once (e.g. from cron):
    python scripts/collect.py
Keep running and check every 30 minutes (leave the terminal open):
    python scripts/collect.py --every 30
Only some companies / skip document downloads:
    python scripts/collect.py --tickers NVDA AAPL --no-docs

Build the database first with scripts/build_db.py; a company that isn't loaded yet
gets a "baseline" load on its first run (history only, no events).
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import config  # noqa: E402
from finsight.datastore import api, collector, db  # noqa: E402
from finsight.datastore import transcript_store  # noqa: E402
from finsight.tools import transcripts as tx  # noqa: E402

LOCK_STALE_SECONDS = 2 * 60 * 60  # a lock older than this is left over from a crash
_indexed_once = False             # first run of this process always indexes (catch-up)


def acquire_lock(path: Path) -> bool:
    """Stop two runs from overlapping (e.g. cron starts one while the last is still going)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def fetch_transcripts(conn, tickers: list[str], args) -> int:
    """New earnings-call transcripts for the newest quarters. Returns how many were stored.

    Never fails the run: a missing key, the daily limit or a network error is printed and skipped.
    """
    try:
        out = transcript_store.ingest_transcripts(conn, tickers, quarters=args.transcript_quarters,
                                                  max_calls=args.transcript_calls)
    except tx.TranscriptAccessError as e:      # no key / bad key / premium
        print(f"  Transcripts skipped: {e}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"  Transcripts failed: {type(e).__name__}: {e}")
        return 0
    rs = out["results"]
    ok = [r for r in rs if r["status"] == "ok"]
    errors = [r for r in rs if r["status"] == "error"]
    print(f"  Transcripts: {out['calls']} calls · {len(ok)} new · "
          f"{sum(1 for r in rs if r['status'] == 'empty')} not published yet · {len(errors)} errors")
    for r in ok:
        print(f"    + {r['ticker']:<6} {r['quarter']}  {r['turns']} turns, {r['words']:,} words")
    for r in errors:
        print(f"    ✗ {r['ticker']:<6} {r['quarter']}  {r['error']}")
    if out["stopped"]:
        print(f"    stopped early: {out['stopped']} (next run continues)")
    return len(ok)


def index_new(conn, tickers: list[str]) -> None:
    """Embed new filings and transcripts into the vector DB so retrieve() can find them.

    Already-indexed ones are skipped. Imported here: loading the embedding model is slow,
    and --no-index runs shouldn't pay for it.
    """
    from finsight.rag.index import index_all, index_transcripts
    from finsight.rag.store import get_store

    try:
        store = get_store()
        res = index_all(conn, tickers, store=store) + index_transcripts(conn, tickers, store=store)
    except Exception as e:  # noqa: BLE001 — e.g. Qdrant unreachable: keep the SEC data, retry next run
        print(f"  RAG index failed: {type(e).__name__}: {e}")
        return
    new = [r for r in res if "passages" in r]
    errors = [r for r in res if "error" in r]
    print(f"  RAG index: {len(new)} new documents ({sum(r['passages'] for r in new):,} paragraphs) · "
          f"{len(errors)} errors")
    for r in errors[:10]:
        print(f"    ✗ {r['accession']}: {r['error']}")
    if len(errors) > 10:
        print(f"    ... {len(errors) - 10} more")


def run_once(args) -> int:
    global _indexed_once
    lock = Path(args.db).parent / "collect.lock"
    if not acquire_lock(lock):
        print(f"Another collection run is in progress ({lock}); skipping.")
        return 0
    try:
        conn = db.connect(args.db)
        tickers = [t.upper() for t in (args.tickers or config.load_companies())]
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Checking {len(tickers)} companies for new filings...")
        results = collector.collect(conn, tickers, years=args.years,
                                    download_docs=not args.no_docs)
        failed = 0
        new_filings = 0
        for r in results:
            if "error" in r:
                failed += 1
                print(f"  ✗ {r['ticker']:<6} {r['error']}")
            elif r["baseline"] and r["new_filings"]:
                new_filings += len(r["new_filings"])
                print(f"  ✓ {r['ticker']:<6} first load: {len(r['new_filings'])} filings (no events)")
            elif r["new_filings"]:
                new_filings += len(r["new_filings"])
                print(f"  ★ {r['ticker']:<6} {len(r['new_filings'])} new filing(s), {r['events']} event(s)")
                for f in r["new_filings"]:
                    print(f"           {f}")
            else:
                print(f"  · {r['ticker']:<6} nothing new")

        # Transcripts run AFTER the SEC step: their quarters come from the filings table,
        # so a new earnings 8-K / 10-Q found above is asked for in the same run.
        new_transcripts = 0 if args.no_transcripts else fetch_transcripts(conn, tickers, args)

        # Index when something is new, and once per process start (catches up after a failed run,
        # e.g. Qdrant was down). Without new content this is skipped, so idle runs stay cheap.
        if not args.no_index and (new_filings or new_transcripts or not _indexed_once):
            index_new(conn, tickers)
            _indexed_once = True

        pending = api.get_pending_events(limit=1000, conn=conn)
        print(f"Pending events for the orchestrator: {len(pending)}")
        return 1 if failed else 0
    finally:
        lock.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", help="default: config/companies.yaml")
    ap.add_argument("--every", type=float, metavar="MINUTES",
                    help="keep running, checking every MINUTES (min 10). Ctrl+C to stop")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--no-docs", action="store_true", help="skip downloading filing documents")
    ap.add_argument("--no-transcripts", action="store_true", help="skip fetching earnings-call transcripts")
    ap.add_argument("--transcript-quarters", type=int, default=2,
                    help="newest N fiscal quarters to keep up to date (default 2)")
    ap.add_argument("--transcript-calls", type=int, default=10,
                    help="max Alpha Vantage calls per run (free key: 25/day)")
    ap.add_argument("--no-index", action="store_true", help="skip updating the RAG vector index")
    ap.add_argument("--db", default=str(config.DB_PATH))
    args = ap.parse_args()

    if not args.every:
        return run_once(args)
    minutes = max(args.every, 10)  # be polite to the SEC
    print(f"Collecting every {minutes:g} minutes. Press Ctrl+C to stop.")
    try:
        while True:
            run_once(args)
            time.sleep(minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
