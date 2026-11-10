# datastore — local SEC database (D3)

Our own copy of the SEC data, so answering a user never calls the SEC live.

## Build it
```
cp .env.example .env            # put your email in SEC_USER_AGENT
pip install -r requirements.txt
python scripts/build_db.py --fresh          # all companies in config/companies.yaml
python scripts/check_quality.py             # sums, gaps, missing metrics
python scripts/demo_sprint1.py NVDA         # Week 5 demo
```
Quick test first: `python scripts/build_db.py --tickers NVDA --no-docs`.
Running `build_db.py` again later only adds what's new.

## Tables (`data/finsight.db`)
| Table | Holds |
|---|---|
| `companies` | ticker, CIK, name, fiscal year end |
| `filings` | accession, form, dates, fiscal year/quarter, 8-K items, links, local file |
| `financial_facts` | one row per company · metric · period, with fiscal label, value, source |
| `ingest_log` | each loader run per company: counts and errors |

## Functions other workstreams call (`finsight.datastore.api`)
```python
from finsight.datastore import api
api.get_company("NVDA")
api.list_filings("NVDA", "10-Q", limit=4)
api.get_filing_text(accession)                  # cleaned text + source link
api.get_metric("NVDA", "revenue", "quarter", 8) # also "annual", "instant"
api.check_new_filings()                         # added by the latest loader run
```
Metrics: revenue, cost_of_revenue, gross_profit, operating_income, net_income,
rd_expense, eps_diluted, cash, total_assets.

## Rules applied to the figures
- Fiscal labels come from the filing (`Q2 FY2027`), never from the calendar.
- Quarters are three-month figures only; prior-year comparisons and year-to-date figures are dropped.
- If a period was reported more than once (e.g. a 10-Q/A), the first-reported value is kept.
- Q4 = full year − nine months, marked `derived`. Not done for EPS.
- Cash and assets are snapshots on one date (`period_type = "instant"`).

## Known limitations
- Restatements are ignored (first-reported wins).
- Q4 EPS isn't available (per-share figures can't be subtracted).
- Concept names differ by company; add overrides in `config/concepts.yaml`
  when `check_quality.py` reports missing metrics.
- Filing text cleaning is basic; section splitting and 8-K Exhibit 99.1 come in Weeks 6–7.

## Collection job (D4): new filings, automatically
```
python scripts/collect.py              # one run: check all companies, store new filings
python scripts/collect.py --every 30   # keep running, check every 30 minutes (Ctrl+C to stop)
```
Each run, per company:
1. **Cheap check:** one SEC call for the filing list, compared with `filings`. Usually nothing new; done.
2. **If something is new:** the normal loader stores it, downloads the document, refreshes figures.
3. **Event:** one row per new 10-K / 10-Q / 8-K in `filing_events`
   (`annual_report`, `quarterly_report`, `earnings_8k`, `other_8k`).

The first run for a company is a **baseline**: it loads history and raises no events.

### For the orchestrator (P4)
```python
from finsight.datastore import api
for e in api.get_pending_events():
    ...  # e["ticker"], e["event_type"], e["fiscal_label"], e["document_url"]
    api.mark_events_processed([e["id"]])
```
Mark an event processed only after handling it, so a crash doesn't lose it.

### Running it on a schedule (macOS)
`crontab -e`, then add (one line, weekdays every 30 min):
```
*/30 * * * 1-5 cd "/path/to/AI-Stock-Research-and-Earnings-Assistant" && .venv/bin/python scripts/collect.py >> data/collect.log 2>&1
```
The laptop must be awake. If cron can't read `~/Documents`, give `cron` Full Disk Access
(System Settings → Privacy & Security), or use `--every 30` in a terminal for the demo.
A lock file (`data/collect.lock`) stops two runs from overlapping.