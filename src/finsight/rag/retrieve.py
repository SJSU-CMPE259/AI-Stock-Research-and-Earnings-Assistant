"""Find the filing paragraphs that answer a question, ready to cite.

    retrieve(question, ticker)          the one call agents need
    get_section(ticker, topic)          a whole section, e.g. latest Risk Factors
    search(question, ticker, ...)       ranked paragraphs: "hybrid" (default), "dense", "sparse"
                                        or "rerank" (hybrid top 30, re-scored by a cross-encoder)
    guess_topics(question)              which sections a question needs

retrieve() works in four steps:
  1. Work out the section(s) the question needs (the router can pass `topics`).
  2. If those sections fit the token budget, return them whole ("section" mode).
  3. Otherwise, search inside them and keep the best paragraphs ("search" mode).
  4. No topic recognised: search all text in the latest filings ("search" mode).

Paragraphs and vectors live in the vector DB (store.py). The SQLite datastore is
only read for the list of filings. Numbers are not answered here: use
datastore.api.get_metric (XBRL).
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

from finsight.datastore import db
from finsight.rag.rerank import RERANK_CANDIDATES, get_reranker
from finsight.rag.store import SEARCH_MODES, Hit, VectorStore, get_store

TOKENS_PER_WORD = 1.35          # English prose, Mistral tokenizer (rough; see eval workbook)
DEFAULT_BUDGET = 12_000         # evidence tokens per question; fits a 32K local model with room
RETRIEVAL_MODES = (*SEARCH_MODES, "rerank")      # the RQ1 arms
DEFAULT_MODE = os.getenv("FINSIGHT_SEARCH_MODE", "hybrid")   # switch to "rerank" once it measures better
NARRATIVE_FORMS = ("10-K", "10-Q")

# Question words -> section topics.
TOPIC_HINTS: list[tuple[str, str]] = [
    (r"\brisks?\b|risk factor|threat|uncertaint|tariff|sanction|regulat|geopolit|supply chain|"
     r"export|china|trade restriction", "risk_factors"),
    (r"\bmd&a\b|management|margin|drove|driver|explain|why did|outlook|guidance|trend|"
     r"liquidity|capital return|segment|demand|operating expense", "mdna"),
    (r"legal|lawsuit|litigation|proceeding|antitrust|court|settle|investigation", "legal"),
    (r"internal control|material weakness|disclosure control|controls and procedures", "controls"),
    (r"interest rate|foreign exchange|currency|\bfx\b|market risk|hedg", "market_risk"),
    (r"cyber|security incident|data breach", "cybersecurity"),
    (r"business model|competit|product (line|portfolio|offering)|customers?|employees|headcount|"
     r"strategy|what does .* (do|sell|make)", "business"),
    (r"critical accounting|accounting estimate|accounting polic", "mdna"),
    (r"press release|earnings release|announce", "press_release"),
]
SUMMARY_RE = re.compile(r"summari|overview|outline|main risks|key risks|list (the|all)|all (the )?risk",
                        re.IGNORECASE)


@dataclass
class Evidence:
    passage_id: str
    ticker: str
    form: str
    fiscal_label: str | None
    filing_date: str
    section: str            # e.g. "Part II Item 1A · Risk Factors"
    topic: str
    heading: str
    text: str
    source_url: str
    n_words: int
    score: float | None = None   # higher = better (cosine / BM25 / RRF); None in section mode

    def citation(self) -> str:
        """'AAPL 10-Q Q3 FY2026 · Part II Item 1A · Risk Factors (filed 2026-08-01) <url>'"""
        when = self.fiscal_label or self.filing_date
        return f"{self.ticker} {self.form} {when} · {self.section} (filed {self.filing_date}) {self.source_url}"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def accession(self) -> str:
        return self.passage_id.split("/", 1)[0]


def _evidence(p: dict, score: float | None = None) -> Evidence:
    return Evidence(
        passage_id=p["passage_id"], ticker=p["ticker"], form=p["form"],
        fiscal_label=p.get("fiscal_label"), filing_date=p["filing_date"], section=p["section"],
        topic=p["topic"], heading=p.get("heading") or "", text=p["text"],
        source_url=p.get("document_url") or "", n_words=p["n_words"], score=score)


def to_cavm(evidence: list[Evidence], prefix: str = "S") -> list[dict]:
    """Label evidence S1, S2... for shared memory, e.g. put("evidence.filing_passages", ...).

    The Report agent cites these labels; passage_id + source_url let the UI link
    each [S#] back to the exact paragraph and filing.
    """
    return [{"id": f"{prefix}{i}", "type": "filing_passage", "text": e.text,
             "citation": e.citation(), **e.to_dict()} for i, e in enumerate(evidence, 1)]


def guess_topics(question: str) -> list[str]:
    q = question.lower()
    out: list[str] = []
    for pattern, topic in TOPIC_HINTS:
        if re.search(pattern, q) and topic not in out:
            out.append(topic)
    return out


def est_tokens(n_words: int) -> int:
    return round(n_words * TOKENS_PER_WORD)


def latest_accessions(ticker: str, forms=NARRATIVE_FORMS, n: int = 1, topic: str | None = None,
                      conn=None, store: VectorStore | None = None) -> list[str]:
    """Newest INDEXED filings for a ticker (optionally only ones that have `topic`)."""
    c, st = conn or db.connect(), store or get_store()
    rows = c.execute(
        f"SELECT accession FROM filings WHERE ticker=? AND form IN ({','.join('?' * len(forms))}) "
        f"AND local_path IS NOT NULL ORDER BY filing_date DESC", [ticker.upper(), *forms]).fetchall()
    out = []
    for (acc,) in rows:
        if st.count(accessions=[acc], topics=[topic] if topic else None):
            out.append(acc)
            if len(out) >= n:
                break
    return out


def get_section(ticker: str, topic: str, forms=NARRATIVE_FORMS, n_filings: int = 1,
                conn=None, store: VectorStore | None = None) -> list[Evidence]:
    """Every paragraph of `topic` from the newest filing(s) that have it, in reading order.

    10-Q risk factors often just say "no material changes" from the 10-K. In that
    case the latest 10-K's section is added, so the answer has real content.
    """
    c, st = conn or db.connect(), store or get_store()
    out: list[Evidence] = []
    for acc in latest_accessions(ticker, forms, n_filings, topic, c, st):
        part = [_evidence(p) for p in st.scroll(accessions=[acc], topics=[topic])]
        short = sum(e.n_words for e in part) < 200
        if short and part and part[0].form == "10-Q" and re.search(
                r"no material change", " ".join(e.text for e in part), re.I):
            part += get_section(ticker, topic, ("10-K",), 1, c, st)
        out += part
    return out


def _dedupe(hits: list[Evidence]) -> list[Evidence]:
    """Drop repeated paragraphs (a 10-Q often repeats the last one's risk text word for word).
    Keeps the best-ranked position, with the newest filing's copy."""
    out: list[Evidence] = []
    seen: dict[str, int] = {}
    for e in hits:
        key = re.sub(r"[^a-z0-9]+", " ", e.text.lower()).strip()[:400]
        if key in seen:
            kept = out[seen[key]]
            if e.filing_date > kept.filing_date:
                e.score = kept.score
                out[seen[key]] = e
            continue
        seen[key] = len(out)
        out.append(e)
    return out


def search(question: str, ticker: str | None = None, topics: list[str] | None = None,
           accessions: list[str] | None = None, k: int = 8, mode: str = DEFAULT_MODE,
           store: VectorStore | None = None, dedupe: bool = True) -> list[Evidence]:
    """Best-matching paragraphs, best first, without repeats.

    mode: "hybrid" (dense + BM25, fused with reciprocal rank fusion), "dense", "sparse",
          or "rerank" (the top RERANK_CANDIDATES hybrid results re-scored by a cross-encoder).
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
    st = store or get_store()
    base = "hybrid" if mode == "rerank" else mode
    n = max(RERANK_CANDIDATES, k * 3) if mode == "rerank" else (k * 3 if dedupe else k)
    hits: list[Hit] = st.search(question, mode=base, k=n, ticker=ticker, topics=topics,
                                accessions=accessions)
    ev = [_evidence(h.payload, h.score) for h in hits]
    if dedupe:
        ev = _dedupe(ev)
    if mode == "rerank" and ev:
        scores = get_reranker().score(question, [f"{e.heading}\n{e.text}" if e.heading else e.text for e in ev])
        for e, sc in zip(ev, scores):
            e.score = sc
        ev.sort(key=lambda e: e.score, reverse=True)
    return ev[:k]


def _pin_short_latest(whole: list[Evidence], max_tokens: int = 1500) -> list[Evidence]:
    """Keep the newest filing's section when it is short (e.g. a 10-Q saying "no material
    changes since the 10-K"): it is the direct answer, so it must not be searched away."""
    if not whole:
        return []
    first = [e for e in whole if e.accession == whole[0].accession]
    if len(first) < len(whole) and sum(est_tokens(e.n_words) for e in first) <= max_tokens:
        return first
    return []


def _fill(evidence: list[Evidence], budget: int) -> list[Evidence]:
    out, used = [], 0
    for e in evidence:
        t = est_tokens(e.n_words)
        if used + t > budget and out:
            break
        out.append(e)
        used += t
    return out


def _result(mode: str, topics: list[str], evidence: list[Evidence], search_mode: str,
            truncated: bool = False) -> dict:
    return {"mode": mode, "search_mode": search_mode if mode == "search" else None,
            "topics": topics, "evidence": evidence, "truncated": truncated,
            "tokens": sum(est_tokens(e.n_words) for e in evidence)}


def retrieve(question: str, ticker: str, topics: list[str] | None = None,
             forms=NARRATIVE_FORMS, n_filings: int = 1, budget_tokens: int = DEFAULT_BUDGET,
             k: int = 8, search_mode: str = DEFAULT_MODE, conn=None,
             store: VectorStore | None = None) -> dict:
    """Evidence for one question about one company.

    topics:      from the router; guessed from the question if not given.
    n_filings:   1 = latest filing; 4 = "over the last four quarters".
    search_mode: "hybrid", "dense", "sparse" or "rerank" when a search is needed (RQ1 ablation).
    Returns {"mode", "search_mode", "topics", "evidence": [Evidence], "tokens", "truncated"}:
      mode "section"  whole section(s) in reading order ("truncated" if a summary
                      question's section was cut at the budget)
           "search"   best-matching paragraphs, best first
           "none"     no indexed filing of these forms for this ticker
    """
    if search_mode not in RETRIEVAL_MODES:
        raise ValueError(f"search_mode must be one of {RETRIEVAL_MODES}")
    c, st = conn or db.connect(), store or get_store()
    topics = topics if topics is not None else guess_topics(question)
    if topics:
        whole = [e for t in topics for e in get_section(ticker, t, forms, n_filings, c, st)]
        if whole:
            if sum(est_tokens(e.n_words) for e in whole) <= budget_tokens:
                return _result("section", topics, whole, search_mode)
            if SUMMARY_RE.search(question):     # a summary needs the section's shape, not a search
                return _result("section", topics, _fill(whole, budget_tokens), search_mode, True)
            # Search inside exactly the filings get_section chose (incl. a 10-K it pulled in).
            accs = list(dict.fromkeys(e.accession for e in whole))
            pinned = _pin_short_latest(whole)
            ids = {e.passage_id for e in pinned}
            hits = search(question, ticker, topics, accs, k=max(k, 40), mode=search_mode, store=st)
            hits = pinned + [e for e in hits if e.passage_id not in ids][:k]
            if len(hits) == len(pinned):    # nothing matched: read the section from the top
                hits = pinned + [e for e in whole if e.passage_id not in ids]
            return _result("search", topics, _fill(hits, budget_tokens), search_mode)
        # topic not in these forms (e.g. an 8-K has no Risk Factors): search all their text

    accs = latest_accessions(ticker, forms, n_filings, conn=c, store=st)
    if not accs:
        return _result("none", topics, [], search_mode)
    hits = search(question, ticker, None, accs, k=k, mode=search_mode, store=st)
    return _result("search", topics, _fill(hits, budget_tokens), search_mode)
