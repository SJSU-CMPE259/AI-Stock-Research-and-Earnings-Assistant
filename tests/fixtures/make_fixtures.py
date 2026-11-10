"""Generate the sample SEC files used by the tests (no internet needed).

    python tests/fixtures/make_fixtures.py

NVIDIA's real fiscal calendar (year ends late January), MADE-UP numbers.
Built-in traps the code must handle:
  - each 10-Q repeats last year's quarter (tagged with the NEW filing's fy/fp)
  - 10-Qs also carry year-to-date figures
  - a 10-Q/A restates FY2026 Q2 later (we must keep the first-reported value)
  - older filings sit on a second submissions page; Form 4s add noise
"""

import json
from pathlib import Path

OUT = Path(__file__).parent
CIK = "0001045810"

# fiscal year -> start, quarter ends (Q1..Q3), year end
CAL = {
    2025: ("2024-01-29", ["2024-04-28", "2024-07-28", "2024-10-27"], "2025-01-26"),
    2026: ("2025-01-27", ["2025-04-27", "2025-07-27", "2025-10-26"], "2026-01-25"),
    2027: ("2026-01-26", ["2026-04-26", "2026-07-26"], None),
}
# quarterly revenue in $ (Q4 only appears via the 10-K full year)
REV = {
    2025: [26_044_000_000, 30_040_000_000, 35_082_000_000, 39_331_000_000],
    2026: [44_062_000_000, 46_743_000_000, 57_006_000_000, 68_189_000_000],
    2027: [81_613_000_000, 96_221_000_000],
}
SHARES = 24_500_000_000


def day_after(d):
    from datetime import date, timedelta
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()


def plus_days(d, n):
    from datetime import date, timedelta
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()


def q_start(fy, qi):
    start, ends, _ = CAL[fy]
    return start if qi == 0 else day_after(ends[qi - 1])


# filings: (accession, form, fy, fp, period_end, filed)
filings = []
n = 100
for fy in (2025, 2026, 2027):
    start, ends, fy_end = CAL[fy]
    for qi, end in enumerate(ends):
        n += 1
        filings.append((f"0001045810-{str(fy - 1)[2:]}-{n:06d}", "10-Q", fy, f"Q{qi + 1}", end,
                        plus_days(end, 30)))
    if fy_end:
        n += 1
        filings.append((f"0001045810-{str(fy)[2:]}-{n:06d}", "10-K", fy, "FY", fy_end,
                        plus_days(fy_end, 31)))
AMEND = ("0001045810-25-000999", "10-Q/A", 2026, "Q2", "2025-07-27", "2025-12-15")
filings.append(AMEND)


def metric_rows(scale=1.0, eps=False):
    """companyfacts rows for one metric, as each filing would report them."""
    rows = []

    def val(fy, qis):
        v = sum(REV[fy][i] for i in qis) * scale
        return round(v / SHARES, 2) if eps else round(v)

    for accn, form, fy, fp, end, filed in filings:
        base = dict(accn=accn, fy=fy, fp=fp, form=form, filed=filed)
        if form == "10-Q/A":  # restated Q2 FY2026 (should be ignored: first reported wins)
            rows.append({**base, "start": q_start(fy, 1), "end": end,
                         "val": val(fy, [1]) + (0.05 if eps else 257_000_000)})
            continue
        if form == "10-Q":
            qi = int(fp[1]) - 1
            rows.append({**base, "start": q_start(fy, qi), "end": end, "val": val(fy, [qi])})
            if fy - 1 in CAL:  # prior-year comparative, tagged with THIS filing's fy/fp
                rows.append({**base, "start": q_start(fy - 1, qi), "end": CAL[fy - 1][1][qi],
                             "val": val(fy - 1, [qi])})
            if qi > 0:  # year-to-date
                rows.append({**base, "start": CAL[fy][0], "end": end,
                             "val": val(fy, list(range(qi + 1)))})
        if form == "10-K":
            rows.append({**base, "start": CAL[fy][0], "end": end, "val": val(fy, [0, 1, 2, 3])})
            if fy - 1 in CAL:
                rows.append({**base, "start": CAL[fy - 1][0], "end": CAL[fy - 1][2],
                             "val": val(fy - 1, [0, 1, 2, 3])})
    return rows


def assets_rows():
    rows = []
    for i, (accn, form, fy, fp, end, filed) in enumerate(filings):
        if form == "10-Q/A":
            continue
        rows.append(dict(accn=accn, fy=fy, fp=fp, form=form, filed=filed, end=end,
                         val=100_000_000_000 + i * 5_000_000_000))
        # balance sheets also show the last year-end (a comparative)
        rows.append(dict(accn=accn, fy=fy, fp=fp, form=form, filed=filed,
                         end=CAL[fy - 1][2] if fy - 1 in CAL else "2024-01-28",
                         val=90_000_000_000))
    return rows


companyfacts = {
    "cik": int(CIK), "entityName": "SAMPLE NVIDIA (made-up values)",
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": metric_rows()}},
        "CostOfRevenue": {"units": {"USD": metric_rows(0.27)}},
        "NetIncomeLoss": {"units": {"USD": metric_rows(0.55)}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": metric_rows(0.55, eps=True)}},
        "Assets": {"units": {"USD": assets_rows()}},
    }},
}


def submissions_block(rows):
    keys = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument", "items"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


sub_rows = [(a, filed, end, form, f"nvda-{a[-6:]}.htm", "") for a, form, _fy, _fp, end, filed in filings]
sub_rows += [
    ("0001045810-26-000500", "2026-08-20", "2026-08-20", "8-K", "nvda-8k-q2.htm", "2.02,9.01"),
    ("0001045810-26-000400", "2026-05-20", "2026-05-20", "8-K", "nvda-8k-q1.htm", "2.02,9.01"),
    ("0001045810-26-000450", "2026-06-25", "2026-06-25", "8-K", "nvda-8k-vote.htm", "5.07"),
    ("0001045810-22-000050", "2022-08-31", "2022-07-31", "10-Q", "old.htm", ""),  # too old
]
sub_rows += [(f"0001045810-26-9{i:05d}", "2026-09-01", "", "4", "f4.xml", "") for i in range(5)]
sub_rows.sort(key=lambda r: r[1], reverse=True)
recent = [r for r in sub_rows if r[1] >= "2025-06-01"]
older = [r for r in sub_rows if r[1] < "2025-06-01"]

submissions = {
    "cik": CIK.lstrip("0"), "name": "NVIDIA CORP", "tickers": ["NVDA"], "fiscalYearEnd": "0125",
    "filings": {
        "recent": submissions_block(recent),
        "files": [{"name": f"CIK{CIK}-submissions-001.json", "filingCount": len(older),
                   "filingFrom": older[-1][1], "filingTo": older[0][1]}],
    },
}
company_tickers = {"0": {"cik_str": int(CIK), "ticker": "NVDA", "title": "NVIDIA CORP"}}

if __name__ == "__main__":
    for name, data in [("nvda_companyfacts.json", companyfacts),
                       ("nvda_submissions.json", submissions),
                       ("nvda_submissions_001.json", submissions_block(older)),
                       ("company_tickers.json", company_tickers)]:
        (OUT / name).write_text(json.dumps(data, indent=1))
        print("wrote", name)
