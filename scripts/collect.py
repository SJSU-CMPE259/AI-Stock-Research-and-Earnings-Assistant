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

LOCK_STALE_SECONDS = 2 * 60 * 60  # a lock older than this is left over from a crash


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


def run_once(args) -> int:
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
        for r in results:
            if "error" in r:
                failed += 1
                print(f"  ✗ {r['ticker']:<6} {r['error']}")
            elif r["baseline"] and r["new_filings"]:
                print(f"  ✓ {r['ticker']:<6} first load: {len(r['new_filings'])} filings (no events)")
            elif r["new_filings"]:
                print(f"  ★ {r['ticker']:<6} {len(r['new_filings'])} new filing(s), {r['events']} event(s)")
                for f in r["new_filings"]:
                    print(f"           {f}")
            else:
                print(f"  · {r['ticker']:<6} nothing new")
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
