"""Earnings-call transcripts (D7): check free access, then fetch and store them.

    python scripts/ingest_transcripts.py --check NVDA                 # 1 call: does the free key work?
    python scripts/ingest_transcripts.py --check NVDA --quarter 2027Q2
    python scripts/ingest_transcripts.py                              # last 8 quarters, every company
    python scripts/ingest_transcripts.py --tickers NVDA AAPL --quarters 4
    python scripts/build_rag_index.py                                 # then index them for RAG

Needs ALPHAVANTAGE_API_KEY in .env (free: https://www.alphavantage.co/support/#api-key) and
the filings database (scripts/build_db.py), which supplies each company's fiscal quarters.
The free key allows 25 calls a day: a run stops cleanly at the limit, and the next run
continues where it left off (stored quarters are skipped).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import config  # noqa: E402
from finsight.datastore import db  # noqa: E402
from finsight.datastore import transcript_store as store  # noqa: E402
from finsight.tools import transcripts as tx  # noqa: E402

QUARTER_WORDS = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th|q[1-4])[- ](fiscal )?quarter(?: of)?(?: fiscal)?(?: year)?"
    r"(?: (?:fiscal )?(?:20)?\d{2,4})?|\bfiscal (?:year )?(?:20)?\d{2,4}\b", re.IGNORECASE)


def check(ticker: str, quarter: str | None) -> int:
    """One call, printed in detail, to confirm free access and the quarter format."""
    if not quarter:
        qs = store.quarters_to_fetch(db.connect(), ticker, 2)
        if not qs:
            print(f"No fiscal quarters for {ticker} in the database: run scripts/build_db.py, "
                  f"or pass --quarter 2026Q2")
            return 1
        # the newest quarter's call may not be published yet: use the one before it
        quarter = qs[1]["quarter"] if len(qs) > 1 else qs[0]["quarter"]
    print(f"Asking {tx.SOURCE} for {ticker} {quarter} (uses 1 of the 25 free daily calls)...")
    try:
        t = tx.get_transcript(ticker, quarter)
    except (tx.TranscriptAccessError, tx.TranscriptLimitError) as e:
        print(f"\nNOT OK: {e}")
        if "premium" in str(e).lower():
            print("-> Transcripts are not on the free plan for this key. Don't ingest; "
                  "keep earnings coverage from 8-K press releases (already indexed).")
        return 1
    if not t.available:
        print(f"\nThe key works, but there is no transcript for {quarter}. Try an older --quarter.")
        return 1
    print(f"\nOK: free access works. {len(t.turns)} speaker turns, {t.n_words:,} words.\n")
    for turn in t.turns[:4]:
        s = f"  sentiment {turn.sentiment:+.2f}" if turn.sentiment is not None else ""
        print(f"  {turn.speaker} ({turn.title}){s}\n    {turn.content[:220]}...\n")
    opening = " ".join(x.content for x in t.turns[:3])
    named = sorted({m.group(0) for m in QUARTER_WORDS.finditer(opening)}, key=len)[:6]
    fy, fp = quarter[:4], "Q" + quarter[-1]
    print(f"Quarter check: we asked for {quarter} = {fp} FY{fy} in our fiscal labels.")
    print(f"  The call's opening mentions: {', '.join(named) if named else '(no quarter phrase found)'}")
    print("  If that names a different quarter, Alpha Vantage's quarter isn't the fiscal one for this "
          "company: don't run the full ingest yet; the mapping in quarter_key() needs a fix.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="TICKER", help="one test call: confirm access and the quarter format")
    ap.add_argument("--quarter", help='with --check: e.g. "2027Q2" (default: the second-newest quarter)')
    ap.add_argument("--tickers", nargs="+", help="default: config/companies.yaml")
    ap.add_argument("--quarters", type=int, default=8, help="newest N fiscal quarters per company")
    ap.add_argument("--max-calls", type=int, default=25, help="stop after this many API calls")
    ap.add_argument("--force", action="store_true", help="fetch again even if stored")
    a = ap.parse_args()

    if a.check:
        return check(a.check.upper(), a.quarter)

    conn = db.connect()
    tickers = [t.upper() for t in (a.tickers or config.load_companies())]
    print(f"Transcripts for {len(tickers)} companies, newest {a.quarters} quarters each "
          f"(max {a.max_calls} calls this run)...")
    try:
        out = store.ingest_transcripts(conn, tickers, a.quarters, a.max_calls, a.force)
    except tx.TranscriptAccessError as e:
        print(f"NOT OK: {e}")
        return 1
    rs = out["results"]
    for r in rs:
        if r["status"] == "skipped":
            continue
        extra = (f"{r['turns']:3} turns {r['words']:6,} words" if r["status"] == "ok"
                 else r.get("error", "no transcript published (asked again tomorrow)"))
        print(f"  {r['ticker']:<6} {r['quarter']}  {r['status']:<6} {extra}")
    n = {s: sum(1 for r in rs if r["status"] == s) for s in ("ok", "empty", "skipped", "error")}
    print(f"\n{out['calls']} calls · {n['ok']} stored · {n['empty']} not published · "
          f"{n['skipped']} already stored · {n['error']} errors")
    if out["stopped"]:
        print(f"Stopped early: {out['stopped']}\nRun again tomorrow to continue where this left off.")
    print("Next: python scripts/build_rag_index.py   (indexes the new transcripts)")
    return 1 if n["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
