#!/usr/bin/env python
"""CLI to fetch and parse SEC filings."""
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.tools.edgar import get_edgar
from backend.app.tools.sections import parse_filing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_nvda():
    """Fetch NVDA's latest 10-K and show sections."""
    edgar = get_edgar()

    print("\n" + "=" * 70)
    print("FinSight: SEC Data Fetcher")
    print("=" * 70)

    # Fetch latest 10-K
    print("\n📄 Fetching NVDA's latest 10-K...")
    filings_10k = edgar.list_filings("NVDA", form="10-K", limit=1)

    if not filings_10k:
        print("❌ No 10-K found")
        return

    filing_10k = filings_10k[0]
    print(f"✅ Found: {filing_10k.form} filed on {filing_10k.filing_date}")
    print(f"   Report date: {filing_10k.report_date}")
    print(f"   URL: {filing_10k.url}")

    # Download and parse
    print("\n📥 Downloading filing...")
    html = edgar.download_filing(filing_10k)
    print(f"✅ Downloaded {len(html):,} bytes")

    print("\n📊 Extracting sections...")
    sections = parse_filing(html, filing_10k.form)
    print(f"✅ Extracted {len(sections)} sections:\n")

    for i, section in enumerate(sections, 1):
        preview = section.text[:150].replace("\n", " ") + "..."
        print(f"   {i}. {section.name}")
        print(f"      {preview}")
        print(f"      ({len(section.text):,} chars)")
        print()

    # Summary
    print("=" * 70)
    print(f"✅ Success! Extracted {len(sections)} sections from {filing_10k.form}")
    print("=" * 70)


if __name__ == "__main__":
    fetch_nvda()
