"""Offline tests: no SEC calls, so they run in CI."""

import json
from pathlib import Path

from finsight.tools.xbrl import extract_metric

FACTS = json.loads(
    (Path(__file__).parent / "fixtures" / "companyfacts_sample.json").read_text()
)


def test_quarterly_uses_fiscal_labels_not_calendar():
    rows = extract_metric(FACTS, "NVDA", "revenue", "quarterly")
    assert [r.label for r in rows] == ["FY2026 Q3", "FY2025 Q3"]
    assert rows[0].value == 57006000000


def test_prior_year_comparative_not_mislabeled():
    # The FY2026 Q3 10-Q repeats last year's Q3 (35.08B) tagged fy=2026.
    # It must come out labeled FY2025 Q3, from its own filing.
    rows = extract_metric(FACTS, "NVDA", "revenue", "quarterly")
    old = [r for r in rows if r.end == "2024-10-27"]
    assert len(old) == 1
    assert old[0].label == "FY2025 Q3"
    assert old[0].accession == "0001045810-24-000316"


def test_quarterly_excludes_year_to_date_values():
    rows = extract_metric(FACTS, "NVDA", "revenue", "quarterly")
    assert 147811000000 not in [r.value for r in rows]  # 9-month YTD


def test_annual():
    rows = extract_metric(FACTS, "NVDA", "revenue", "annual")
    assert [(r.label, r.value) for r in rows] == [("FY2025 FY", 130497000000)]


def test_point_in_time_metric():
    rows = extract_metric(FACTS, "NVDA", "total_assets", "quarterly")
    assert [(r.label, r.end) for r in rows] == [("FY2026 Q3", "2025-10-26")]


def test_unknown_metric_returns_empty():
    assert extract_metric(FACTS, "NVDA", "NotARealConcept") == []
