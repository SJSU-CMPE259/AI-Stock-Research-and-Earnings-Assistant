import sys
from pathlib import Path
import time

# Add the project root to Python's import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.sec_client import get_json, _cache_path

# Public SEC API endpoint for Apple's company submissions.
url = "https://data.sec.gov/submissions/CIK0000320193.json"



# Test 1: Fetch JSON data.
print("TEST 1: Fetching SEC data")
start = time.time()

data = get_json(url)

print("Company:", data["name"])
print("CIK:", data["cik"])
print("Ticker:", data["tickers"])
print("Time:", round(time.time() - start, 3), "seconds")



# Test 2: Verify the cache file exists.
print("\nTEST 2: Checking cache")
cache_file = _cache_path(url, ".json")
print("Cache path:", cache_file)
print("Cache exists:", cache_file.exists())



# Test 3: Request the same data again.
print("\nTEST 3: Fetching cached data")
start = time.time()

cached_data = get_json(url)

print("Time:", round(time.time() - start, 3), "seconds")
print("Data matches:", data == cached_data)