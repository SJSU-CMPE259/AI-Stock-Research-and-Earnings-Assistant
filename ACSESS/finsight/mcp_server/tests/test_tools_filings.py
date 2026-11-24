"""tools.list_filings and tools.get_filing_section with a fake Edgar (no SEC calls)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_server import tools

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_10k.html"
URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000010/nvda-20260125.htm"


def _filing(accession="0001045810-26-000010", filing_date="2026-02-25"):
    # Same attributes as backend.app.tools.edgar.Filing, without importing the SEC client.
    return SimpleNamespace(ticker="NVDA", cik="0001045810", form="10-K", accession=accession,
                           filing_date=filing_date, report_date="2026-01-25", url=URL,
                           primary_document="nvda-20260125.htm")


class FakeEdgar:
    def __init__(self, filings=None):
        self.filings = [_filing()] if filings is None else filings
        self.list_calls = []
        self.downloads = 0

    def list_filings(self, ticker, form="10-K", limit=10):
        self.list_calls.append((ticker, form, limit))
        return self.filings[:limit]

    def download_filing(self, filing, cache=True):
        self.downloads += 1
        return FIXTURE.read_text()


@pytest.fixture
def edgar(monkeypatch):
    fake = FakeEdgar()
    monkeypatch.setattr(tools, "_edgar", lambda: fake)
    tools._sections_cache.clear()
    yield fake
    tools._sections_cache.clear()


# ---- list_filings ----------------------------------------------------------

def test_list_filings_returns_plain_dicts(edgar):
    out = tools.list_filings("nvda")
    assert edgar.list_calls == [("NVDA", "10-K", 5)]
    assert out["source"] == "SEC EDGAR"
    assert out["filings"] == [{"form": "10-K", "filing_date": "2026-02-25", "report_date": "2026-01-25",
                               "accession": "0001045810-26-000010", "url": URL}]
    json.dumps(out)  # plain, serializable


def test_list_filings_none_found_is_note_not_error(edgar):
    edgar.filings = []
    out = tools.list_filings("NVDA", form="10-q")
    assert edgar.list_calls == [("NVDA", "10-Q", 5)]
    assert out["filings"] == [] and "No 10-Q filings" in out["note"]


# ---- get_filing_section ----------------------------------------------------

@pytest.mark.parametrize("item", ["1a", "1A", "Item 1A", "ITEM 1A.", "Item 1A. Risk Factors", "risk factors"])
def test_filing_section_item_aliases(edgar, item):
    out = tools.get_filing_section("nvda", "10-K", item)
    assert out["item"] == "Item 1A"
    assert out["text"].startswith("Item 1A. Risk Factors")
    assert "Business paragraph" not in out["text"]
    assert out["citation"] == f"[F1] NVDA 10-K filed 2026-02-25, Item 1A ({URL})"
    assert out["filing"] == {"accession": "0001045810-26-000010", "filing_date": "2026-02-25", "url": URL}
    assert out["content_type"] == "external_document_text"


@pytest.mark.parametrize("item, form, expected", [
    ("Item 1 A", "10-K", "Item 1A"),
    ("7 a", "10-K", "Item 7A"),
    ("1 business", "10-K", "Item 1"),
    ("market risk", "10-K", "Item 7A"),
    ("market risk", "10-Q", "Item 3"),   # 10-Q Part I Item 3
    ("md&a", "10-K", "Item 7"),
    ("MD&A", "10-Q", "Item 2"),
])
def test_normalize_item_spacing_and_form_aware_names(item, form, expected):
    assert tools.normalize_item(item, form) == expected


def test_normalize_item_name_not_in_form_is_error():
    with pytest.raises(ValueError, match="10-Q"):
        tools.normalize_item("business", "10-Q")


@pytest.mark.parametrize("kwargs", [{"max_chars": 0}, {"max_chars": 50_001}, {"offset": -1}])
def test_filing_section_paging_args_out_of_range_rejected(edgar, kwargs):
    with pytest.raises(ValueError, match="max_chars|offset"):
        tools.get_filing_section("NVDA", "10-K", "1A", **kwargs)
    assert edgar.list_calls == [] and edgar.downloads == 0


def test_filing_section_paging_offsets_and_total_chars(edgar):
    first = tools.get_filing_section("NVDA", "10-K", "Item 1A", max_chars=1000)
    total = first["total_chars"]
    assert total > 2000
    assert (first["offset"], first["returned_chars"], first["next_offset"]) == (0, 1000, 1000)

    pages, offset = [first["text"]], first["next_offset"]
    while offset is not None:
        page = tools.get_filing_section("NVDA", "10-K", "Item 1A", max_chars=1000, offset=offset)
        assert page["offset"] == offset and page["total_chars"] == total
        pages.append(page["text"])
        offset = page["next_offset"]
    assert page["returned_chars"] == total - page["offset"]
    assert len("".join(pages)) == total
    assert "".join(pages).rstrip().endswith("Risk paragraph 30: this synthetic sentence exists only to give "
                                            "the section enough body text for the parser's 2000-character "
                                            "minimum.")


def test_filing_section_downloads_once_for_multiple_pages(edgar):
    tools.get_filing_section("NVDA", "10-K", "1A", max_chars=500)
    tools.get_filing_section("NVDA", "10-K", "1A", max_chars=500, offset=500)
    tools.get_filing_section("NVDA", "10-K", "Item 7")
    assert edgar.downloads == 1


def test_filing_section_8k_unsupported_error(edgar):
    with pytest.raises(ValueError, match="8-K"):
        tools.get_filing_section("NVDA", "8-K", "Item 2.02")
    assert edgar.list_calls == [] and edgar.downloads == 0


def test_filing_section_unsupported_item_lists_supported(edgar):
    with pytest.raises(ValueError, match="Item 1, Item 1A, Item 7, Item 7A"):
        tools.get_filing_section("NVDA", "10-K", "Item 9")
    assert edgar.downloads == 0


def test_filing_section_missing_item_lists_found_items(edgar):
    with pytest.raises(ValueError, match="found: Item 1, Item 1A, Item 7"):
        tools.get_filing_section("NVDA", "10-K", "7A")


def test_filing_section_offset_past_end_rejected(edgar):
    total = tools.get_filing_section("NVDA", "10-K", "1A")["total_chars"]
    with pytest.raises(ValueError, match="offset"):
        tools.get_filing_section("NVDA", "10-K", "1A", offset=total)


def test_filing_section_no_filings_error(edgar):
    edgar.filings = []
    with pytest.raises(ValueError, match="No 10-K filings found for NVDA"):
        tools.get_filing_section("NVDA", "10-K", "1A")
