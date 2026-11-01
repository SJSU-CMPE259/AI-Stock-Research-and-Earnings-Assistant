"""Tests for finsight.rag: section splitting, vector indexing, retrieval and eval.

No network and no model download: an in-memory Qdrant and the offline HashEmbedder.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

import finsight.rag.store as store_mod

from finsight.datastore import db
from finsight.datastore.api import html_to_text
from finsight.rag import exhibits, index_all, index_filing, retrieve, search, to_cavm
from finsight.rag.retrieve import get_section, guess_topics
from finsight.rag.embeddings import BM25Encoder, HashEmbedder, tokenize
from finsight.rag.eval import evaluate
from finsight.rag.sections import find_items, split_filing
from finsight.rag.store import VectorStore

P = "<p>{}</p>"
LONG = ("Apple's operations and performance depend significantly on global and regional "
        "economic conditions and adverse economic conditions can materially adversely "
        "affect the Company's business, results of operations and financial condition. ")

TENQ_HTML = f"""<html><head><style>.x{{}}</style></head><body>
<div style="display:none"><ix:header>hidden xbrl header Item 7 junk</ix:header></div>
<table>
<tr><td>Part I</td></tr>
<tr><td>Item 1.</td><td>Financial Statements</td><td>1</td></tr>
<tr><td>Item 2.</td><td>Management's Discussion and Analysis</td><td>13</td></tr>
<tr><td>Part II</td></tr>
<tr><td>Item 1.</td><td>Legal Proceedings</td><td>20</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>21</td></tr>
</table>
<div>PART I — FINANCIAL INFORMATION</div>
<div>Item 1. Financial Statements</div>
<table><tr><td>Net sales</td><td>$</td><td>94,036</td><td>$</td><td>85,777</td></tr>
<tr><td>Cost of sales</td><td>$</td><td>50,318</td><td>$</td><td>46,099</td></tr></table>
<p>The accompanying notes are an integral part of these condensed consolidated financial statements.</p>
<div>Apple Inc. | Q3 2026 Form 10-Q | 5</div>
<div>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</div>
<p>Gross Margin</p>
{P.format("Products gross margin increased during the third quarter due to favorable mix and cost savings, partially offset by tariff costs.")}
{P.format("Services gross margin percentage increased primarily due to a different mix of services.")}
<div>Apple Inc. | Q3 2026 Form 10-Q | 14</div>
<div>Item 4. Controls and Procedures</div>
{P.format("There were no changes in the Company's internal control over financial reporting during the quarter.")}
<div>PART II — OTHER INFORMATION</div>
<div>Item 1. Legal Proceedings</div>
{P.format("Epic Games filed a lawsuit alleging violations of federal and state antitrust laws.")}
<div>Item 1A.</div><div>Risk Factors</div>
{P.format("New tariffs and trade restrictions imposed by the U.S. and other countries could adversely affect the Company's supply chain and gross margin.")}
{P.format(LONG * 30)}
<div>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds</div>
{P.format("The Company repurchased shares during the quarter under its program.")}
</body></html>"""

TENK_HTML = f"""<html><body>
<div>PART I</div>
<div>Item 1. Business</div>{P.format("The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The markets for the Company's products are highly competitive.")}
<div>Item 1A. Risk Factors</div>{P.format("The Company faces substantial competition and rapid technological change in artificial intelligence features.")}{P.format(LONG)}
<div>Item 7. Management's Discussion and Analysis</div>{P.format("Critical accounting estimates include uncertain tax positions and legal contingencies.")}
</body></html>"""

QUIET_10Q = f"""<html><body><div>PART II</div><div>Item 1A. Risk Factors</div>
{P.format("There have been no material changes to the Company's risk factors since the 2025 Form 10-K.")}
</body></html>"""


class _Conn:
    """sqlite3 connection plus an add() helper for test filings."""
    def __init__(self, c):
        self._c = c

    def __getattr__(self, name):
        return getattr(self._c, name)


@pytest.fixture
def store(monkeypatch):
    st = VectorStore(QdrantClient(location=":memory:"), dense=HashEmbedder(), collection="test")
    monkeypatch.setattr(store_mod, "_default", st)      # every get_store() call uses it
    return st


@pytest.fixture
def conn(tmp_path, store):
    raw = db.connect(tmp_path / "t.db")
    c = _Conn(raw)
    c.execute("INSERT INTO companies (ticker, cik, name) VALUES ('AAPL','0000320193','Apple Inc.')")

    def add(acc, form, date, html, fy=2026, fp="Q3"):
        path = tmp_path / "filings" / "AAPL" / f"{acc}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        c.execute("""INSERT INTO filings (accession, ticker, form, filing_date, fiscal_year,
                     fiscal_period, document_url, local_path, added_at)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (acc, "AAPL", form, date, fy, fp,
                   f"https://www.sec.gov/Archives/edgar/data/320193/{acc}/doc.htm", str(path), date))
        return path
    c.add = add
    add("0000320193-25-000079", "10-K", "2025-10-31", TENK_HTML, 2025, "FY")
    add("0000320193-26-000020", "10-Q", "2026-08-01", TENQ_HTML)
    c.commit()
    return c


# ---------- splitting ----------

def test_10q_items_found_and_toc_ignored():
    text = html_to_text(TENQ_HTML)
    heads = [(p, i) for _, _, p, i, _ in find_items(text, "10-Q")]
    assert heads == [("I", "1"), ("I", "2"), ("I", "4"), ("II", "1"), ("II", "1A"), ("II", "2")]


def test_sections_have_topics_titles_and_paragraphs():
    secs = {s.topic: s for s in split_filing(html_to_text(TENQ_HTML), "10-Q")}
    assert set(secs) >= {"financial_statements", "mdna", "controls", "legal", "risk_factors"}
    assert secs["risk_factors"].title == "Risk Factors"            # title on the next line
    mdna = secs["mdna"].passages
    assert mdna[0].heading == "Gross Margin" and "favorable mix" in mdna[0].text
    assert not any("Form 10-Q |" in p.text for p in mdna)           # page footer dropped
    fs = secs["financial_statements"].passages
    assert fs[0].kind == "table"                                     # numbers aren't searched


def test_long_paragraphs_are_cut():
    secs = {s.topic: s for s in split_filing(html_to_text(TENQ_HTML), "10-Q")}
    assert max(p.n_words for p in secs["risk_factors"].passages) <= 320
    assert len(secs["risk_factors"].passages) > 3


def test_no_headings_falls_back_to_full_text():
    secs = split_filing("Just some text.\n\nAnd another paragraph of text here.", "10-K")
    assert len(secs) == 1 and secs[0].topic == "full_text"


# ---------- indexing ----------

def test_index_is_idempotent(conn, store):
    first = index_all(conn)
    assert all("passages" in r for r in first)
    n = store.count()
    assert n == sum(r["passages"] for r in first) > 0
    again = index_all(conn)
    assert all(r.get("skipped") for r in again)
    index_filing(conn, "0000320193-26-000020", force=True)
    assert store.count() == n                                      # replaced, not duplicated


def test_passage_ids_are_stable_and_tables_left_out(conn, store):
    index_all(conn)
    payloads = store.scroll()
    ids = {p["passage_id"] for p in payloads}
    assert "0000320193-26-000020/II-1A/p000" in ids
    assert "0000320193-25-000079/1A/p000" in ids
    assert not any("94,036" in p["text"] for p in payloads)        # numbers come from XBRL
    p = next(p for p in payloads if p["topic"] == "legal")
    assert p["fiscal_label"] == "Q3 FY2026" and p["section"] == "Part II Item 1 · Legal Proceedings"


def test_dimension_mismatch_is_reported(conn, store):
    index_all(conn)
    other = VectorStore(store.client, dense=HashEmbedder(dim=64), collection="test")
    with pytest.raises(RuntimeError, match="--rebuild"):
        other.count()


# ---------- retrieval ----------

def test_guess_topics():
    assert guess_topics("What new risks did Apple flag?") == ["risk_factors"]
    assert "mdna" in guess_topics("What drove the change in gross margin?")
    assert guess_topics("Is Apple in any lawsuits?") == ["legal"]
    assert guess_topics("hello") == []
    assert guess_topics("What does Nvidia say about export controls to China?") == ["risk_factors"]
    assert guess_topics("Any material weakness in internal control?") == ["controls"]
    assert guess_topics("Do limits on selling products to some countries matter?") == []
    assert guess_topics("How does Nvidia describe its competition?") == ["business"]


def test_section_mode_returns_whole_section_in_order(conn):
    index_all(conn)
    r = retrieve("What drove the gross margin change?", "AAPL", topics=["mdna"], conn=conn)
    assert r["mode"] == "section"
    assert [e.form for e in r["evidence"]] == ["10-Q", "10-Q"]       # latest filing only
    assert "Part I Item 2" in r["evidence"][0].section
    assert r["evidence"][0].fiscal_label == "Q3 FY2026"


def test_over_budget_switches_to_search(conn):
    index_all(conn)
    r = retrieve("tariffs trade restrictions", "AAPL", topics=["risk_factors"],
                 budget_tokens=300, conn=conn)
    assert r["mode"] == "search"
    assert "tariffs" in r["evidence"][0].text
    assert r["tokens"] <= 300 or len(r["evidence"]) == 1


@pytest.mark.parametrize("mode", ["sparse", "dense", "hybrid"])
def test_every_search_mode_finds_the_tariff_paragraph(conn, mode):
    index_all(conn)
    hits = search("tariffs and trade restrictions", "AAPL", topics=["risk_factors"], k=3, mode=mode)
    assert "tariffs" in hits[0].text
    assert hits[0].score is not None
    assert search("tariffs", "MSFT", mode=mode) == []              # ticker filter


def test_search_filters_by_accession(conn):
    index_all(conn)
    hits = search("artificial intelligence competition", "AAPL",
                  accessions=["0000320193-26-000020"], k=20, mode="hybrid")
    assert hits and all(h.passage_id.startswith("0000320193-26-000020") for h in hits)


def test_bad_mode_rejected(conn):
    with pytest.raises(ValueError):
        retrieve("x", "AAPL", search_mode="fuzzy", conn=conn)


def test_bm25_encoder():
    enc = BM25Encoder()
    assert tokenize("The restrictions on exports") == ["restrict", "export"]   # stemmed, no stopwords
    idx, val = enc.encode_document("tariffs tariffs trade")
    assert len(idx) == 2 and max(val) > min(val)                    # repeated term weighs more
    assert enc.encode_query("tariffs")[0] == [enc.encode_document("tariffs")[0][0]]


def test_evaluate_reports_hit_recall_and_mrr(conn):
    index_all(conn)
    cases = [{"id": "tariffs", "question": "tariffs trade restrictions", "ticker": "AAPL",
              "topics": ["risk_factors"], "gold_regex": "tariff"},
             {"id": "epic", "question": "antitrust lawsuit", "ticker": "AAPL",
              "gold_ids": ["0000320193-26-000020/II-1/p000"]}]
    rows, summary = evaluate(cases, k=5)
    assert len(rows) == 6 and {s["mode"] for s in summary} == {"sparse", "dense", "hybrid"}
    for s in summary:
        assert s["hit@5"] == 1.0 and 0 < s["MRR"] <= 1.0


def test_no_material_changes_pulls_in_10k(conn):
    conn.add("0000320193-26-000030", "10-Q", "2026-10-30", QUIET_10Q, 2026, "Q4")
    conn.commit()
    index_all(conn)
    ev = get_section("AAPL", "risk_factors", conn=conn)
    assert ev[0].form == "10-Q" and "no material changes" in ev[0].text
    assert any(e.form == "10-K" and "artificial intelligence" in e.text for e in ev)


def test_quiet_10q_over_budget_searches_the_10k_too(conn):
    conn.add("0000320193-26-000030", "10-Q", "2026-10-30", QUIET_10Q, 2026, "Q4")
    conn.commit()
    index_all(conn)
    r = retrieve("artificial intelligence competition", "AAPL", topics=["risk_factors"],
                 budget_tokens=60, conn=conn)
    assert r["mode"] == "search"
    assert r["evidence"][0].form == "10-Q"                          # direct answer kept first
    assert any("artificial intelligence" in e.text for e in r["evidence"])


def test_no_filing_of_requested_form_returns_nothing(conn):
    index_all(conn)
    r = retrieve("What did Apple disclose in its 8-Ks?", "AAPL", forms=("8-K",), topics=[], conn=conn)
    assert r["mode"] == "none" and r["evidence"] == []
    r = retrieve("What risks?", "AAPL", forms=("8-K",), conn=conn)
    assert r["mode"] == "none"


def test_summary_question_reads_section_in_order(conn):
    index_all(conn)
    r = retrieve("Summarize the main risks in the latest 10-Q.", "AAPL", forms=("10-Q",),
                 topics=["risk_factors"], budget_tokens=150, conn=conn)
    assert r["mode"] == "section" and r["truncated"]
    assert [e.passage_id for e in r["evidence"]][0].endswith("/II-1A/p000")
    assert r["tokens"] <= 150 or len(r["evidence"]) == 1


def test_to_cavm_labels(conn):
    index_all(conn)
    items = to_cavm(retrieve("legal proceedings", "AAPL", conn=conn)["evidence"])
    assert items[0]["id"] == "S1" and "Epic Games" in items[0]["text"]
    assert items[0]["citation"].startswith("AAPL 10-Q Q3 FY2026 · Part II Item 1 · Legal Proceedings")


# ---------- 8-K press releases ----------

def test_8k_press_release_indexed(conn, monkeypatch, tmp_path):
    acc = "0000320193-26-000019"
    conn.add(acc, "8-K", "2026-07-31",
             "<html><body><div>Item 2.02 Results of Operations and Financial Condition</div>"
             "<p>On July 31, 2026, Apple issued a press release, attached as Exhibit 99.1.</p>"
             "<div>Item 9.01 Financial Statements and Exhibits</div><p>Exhibit 99.1 press release.</p>"
             "</body></html>", 2026, "Q3")
    conn.commit()
    listing = {"directory": {"item": [{"name": "aapl-20260731.htm"},
                                      {"name": "a8-kex991q3202606272026.htm"},
                                      {"name": "a8-kex992q3202606272026.htm"}]}}
    monkeypatch.setattr(exhibits.sec_client, "get_json", lambda url, ttl=None: listing)
    monkeypatch.setattr(exhibits.sec_client, "get_text", lambda url, ttl=None:
                        "<p>Tim Cook said the quarter set a June-quarter revenue record for Services.</p>")
    path = exhibits.download_press_release(conn, acc)
    assert path and path.name == f"{acc}.ex99.html"
    index_filing(conn, acc)
    r = retrieve("What did Apple announce in the earnings press release?", "AAPL",
                 forms=("8-K",), conn=conn)
    assert any("Tim Cook" in e.text for e in r["evidence"])


def test_pick_ex99():
    assert exhibits.pick_ex99(["a.htm", "ex-99_2.htm", "ex-99_1.htm"]) == "ex-99_1.htm"
    assert exhibits.pick_ex99(["a.htm", "R1.htm"]) is None
