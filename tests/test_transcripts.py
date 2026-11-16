"""Tests for earnings-call transcripts: parsing, storing, splitting, indexing and retrieval.

No network: Alpha Vantage is replaced by a fake fetch function.
"""

from __future__ import annotations

import pytest

from finsight import config
from finsight.datastore import transcript_store as tstore
from finsight.rag import retrieve
from finsight.rag.index import index_transcripts
from finsight.rag.retrieve import guess_topics, latest_accessions
from finsight.rag.sections import split_transcript
from finsight.tools import transcripts as tx
from test_rag import conn, store  # noqa: F401  (pytest fixtures)

LONG_ANSWER = ("Demand for our data center platforms remains very strong, and supply is improving "
               "each quarter as we ramp new systems with our partners. ") * 30

RAW = {"symbol": "AAPL", "quarter": "2026Q3", "transcript": [
    {"speaker": "Operator", "title": "Operator", "content": "Good day and welcome to the third quarter "
     "fiscal 2026 earnings conference call.", "sentiment": "0.1"},
    {"speaker": "Tim Cook", "title": "CEO", "content": "Thank you. We delivered record services revenue "
     "and our installed base reached a new all-time high across every geographic segment this quarter, "
     "driven by strong upgrades and customers switching to our products.", "sentiment": "0.8"},
    {"speaker": "Kevan Parekh", "title": "CFO", "content": "Gross margin was 46.5 percent, within our "
     "guidance range, as tariff costs were offset by favorable mix and cost savings across products.",
     "sentiment": "0.5"},
    {"speaker": "Operator", "title": "Operator", "content": "We will now begin the question-and-answer "
     "session. Our first question comes from Erik Woodring.", "sentiment": "0"},
    {"speaker": "Erik Woodring", "title": "Morgan Stanley - Analyst", "content": "How should we think "
     "about AI demand and memory costs next quarter?", "sentiment": "0.2"},
    {"speaker": "Tim Cook", "title": "CEO", "content": LONG_ANSWER, "sentiment": "0.7"},
    {"speaker": "Erik Woodring", "title": "Morgan Stanley - Analyst", "content": "Thank you.",
     "sentiment": "0"},
]}


# ---------- the API tool ----------

def test_parse_transcript_turns_and_sentiment():
    t = tx.parse_transcript("aapl", "2026Q3", RAW)
    assert t.available and t.ticker == "AAPL" and len(t.turns) == 7
    assert t.turns[1].speaker == "Tim Cook" and t.turns[1].sentiment == 0.8
    assert "apikey" not in t.source_url


@pytest.mark.parametrize("body, err", [
    ({"Information": "This is a premium endpoint. Subscribe to any of the premium plans."},
     tx.TranscriptAccessError),
    ({"Information": "We have detected your API key as ... standard API rate limit is 25 requests per day."},
     tx.TranscriptLimitError),
    ({"Error Message": "Invalid API call. Please retry or visit the documentation."},
     tx.TranscriptAccessError),
])
def test_limit_and_access_errors_are_detected(body, err):
    with pytest.raises(err):
        tx.parse_transcript("AAPL", "2026Q3", body)


def test_empty_transcript_is_not_available():
    t = tx.parse_transcript("AAPL", "2027Q1", {"symbol": "AAPL", "quarter": "2027Q1", "transcript": []})
    assert not t.available


def test_quarter_key():
    assert tx.quarter_key(2027, "Q2") == "2027Q2"
    assert tx.quarter_key(2026, "FY") == "2026Q4"
    with pytest.raises(ValueError):
        tx.quarter_key(2026, "H1")


# ---------- splitting ----------

def test_split_transcript_remarks_then_qa_with_speakers():
    secs = split_transcript(tx.parse_transcript("AAPL", "2026Q3", RAW).turns)
    assert [s.title for s in secs] == ["Prepared remarks", "Q&A"]
    assert all(s.topic == "earnings_call" for s in secs)
    qa = secs[1].passages
    assert qa[0].text.startswith("Operator: We will now begin")                 # operator merged in
    assert "Erik Woodring (Morgan Stanley - Analyst): How should we think" in qa[0].text
    assert all(p.n_words <= 340 for p in qa)                                    # long answer cut
    assert sum("Tim Cook (CEO):" in p.text for p in qa) >= 2                    # every chunk attributed


# ---------- storing ----------

def _fake_fetch(calls):
    def fetch(ticker, quarter):
        calls.append((ticker, quarter))
        if quarter == "2026Q3":
            return tx.parse_transcript(ticker, quarter, RAW)
        return tx.Transcript(ticker, quarter)            # not published
    return fetch


def test_quarters_come_from_filings_with_8k_call_date(conn):
    conn.execute("""INSERT INTO filings (accession, ticker, form, filing_date, fiscal_year, fiscal_period,
                    items, added_at) VALUES ('8k-1','AAPL','8-K','2026-07-30',2026,'Q3','2.02,9.01','x')""")
    qs = tstore.quarters_to_fetch(conn, "AAPL")
    assert [q["quarter"] for q in qs] == ["2026Q3", "2025Q4"]                  # 10-K "FY" -> Q4
    assert qs[0]["call_date"] == "2026-07-30" and qs[1]["call_date"] == "2025-10-31"


def test_ingest_saves_skips_and_stops_at_limit(conn, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "x.db")
    calls = []
    out = tstore.ingest_transcripts(conn, ["AAPL"], fetch=_fake_fetch(calls))
    assert {r["quarter"]: r["status"] for r in out["results"]} == {"2026Q3": "ok", "2025Q4": "empty"}
    assert (tmp_path / "transcripts" / "AAPL" / "2026Q3.json").exists()
    out = tstore.ingest_transcripts(conn, ["AAPL"], fetch=_fake_fetch(calls))
    assert out["calls"] == 0                       # ok stored, empty re-asked only after a day

    def limited(ticker, quarter):
        raise tx.TranscriptLimitError("25 requests per day")
    out = tstore.ingest_transcripts(conn, ["AAPL"], force=True, fetch=limited)
    assert out["stopped"] and out["calls"] == 1


# ---------- indexing and retrieval ----------

@pytest.fixture
def with_transcript(conn, store, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "x.db")
    from finsight.rag import index_all
    index_all(conn, store=store)
    tstore.ingest_transcripts(conn, ["AAPL"], fetch=_fake_fetch([]))
    res = index_transcripts(conn, store=store)
    assert res[0]["passages"] >= 3
    return conn


def test_index_transcripts_is_idempotent(with_transcript, store):
    assert index_transcripts(with_transcript, store=store)[0]["skipped"]


def test_guess_topics_earnings_call():
    assert "earnings_call" in guess_topics("What did the CEO say on the earnings call about AI demand?")
    assert "earnings_call" in guess_topics("What did analysts ask in the Q&A?")
    assert "earnings_call" not in guess_topics("What are Apple's main risks?")


def test_earnings_call_question_retrieves_transcript(with_transcript, store):
    r = retrieve("What did analysts ask on the earnings call about AI?", "AAPL",
                 budget_tokens=200, store=store, conn=with_transcript)
    assert r["topics"] == ["earnings_call"]
    assert r["evidence"] and all(e.form == "TRANSCRIPT" for e in r["evidence"])
    top = r["evidence"][0]
    assert top.accession == "TR-AAPL-2026Q3" and top.fiscal_label == "Q3 FY2026"
    assert "Earnings call" in top.citation() and "apikey" not in top.citation()


def test_mixed_question_uses_filing_and_call(with_transcript, store):
    r = retrieve("What did the CEO say on the earnings call about AI demand?", "AAPL",
                 budget_tokens=200, store=store, conn=with_transcript)
    assert set(r["topics"]) == {"mdna", "earnings_call"}                 # "demand" -> MD&A too
    assert any(e.form == "TRANSCRIPT" for e in r["evidence"])


def test_filing_questions_do_not_see_transcripts(with_transcript, store):
    r = retrieve("What new risks did Apple flag about tariffs?", "AAPL", store=store, conn=with_transcript)
    assert r["evidence"] and all(e.form != "TRANSCRIPT" for e in r["evidence"])
    assert "TR-AAPL-2026Q3" not in latest_accessions("AAPL", n=5, conn=with_transcript, store=store)
    assert latest_accessions("AAPL", ("TRANSCRIPT",), 1, conn=with_transcript, store=store) == ["TR-AAPL-2026Q3"]
