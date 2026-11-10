import sys
from pathlib import Path
import time

# Add project root to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsight.tools.edgar import (
    get_cik,
    get_filings,
    get_latest_filing,
    get_filing_html,
)



# TEST 1: Find company's CIK
print("\nTEST 1: Get CIK")
cik = get_cik("NVDA")

print("Ticker: NVDA")
print("CIK:", cik)
assert cik == "0001045810"



# TEST 2: Get recent 10-Q filings
print("\nTEST 2: Get recent 10-Q filings")
filings = get_filings("NVDA", form="10-Q", limit=4)

print("Number of filings:", len(filings))

for filing in filings:
    print("--------------------")
    print("Form:", filing.form)
    print("Filing date:", filing.filing_date)
    print("Period:", filing.period_of_report)
    print("Accession:", filing.accession)
    print("URL:", filing.url)

assert len(filings) > 0
assert len(filings) <= 4
assert all(f.form == "10-Q" for f in filings)



# TEST 3: Get latest 10-K
print("\nTEST 3: Get latest 10-K")
latest = get_latest_filing("NVDA", form="10-K")

print("Ticker:", latest.ticker)
print("Form:", latest.form)
print("Filing date:", latest.filing_date)
print("URL:", latest.url)

assert latest.form == "10-K"
assert latest.url.startswith("https://www.sec.gov/Archives/")



# TEST 4: Download filing HTML
print("\nTEST 4: Download filing HTML")
start = time.time()

html = get_filing_html(latest)

print("HTML length:", len(html))
print("Time:", round(time.time() - start, 3), "seconds")
print("HTML preview:", html[:200])

assert len(html) > 0


# Test 5: Invalid ticker
print("\nTEST 5: Invalid ticker")

try:
    get_cik("INVALID_TICKER_123")
except ValueError as e:
    print("PASSED:", e)
else:
    raise AssertionError("Expected ValueError for invalid ticker")


print("\nAll 5 tests passed!")