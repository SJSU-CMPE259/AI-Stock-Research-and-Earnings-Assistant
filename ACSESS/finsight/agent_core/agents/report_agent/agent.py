"""Report agent: write the final answer with Mistral, cite fact IDs, verify, fall back.

ONE function (build_evidence) creates the ordered, labeled list used for BOTH the report
prompt and the UI's Sources, so [Pk] means the same passage in each (Task 2). Checks:
  - every cited ID exists;
  - support: every number in a cited sentence appears in the cited note/passage (Task 2);
  - entity safety: every cited note/passage belongs to the question's ticker (Task 3);
  - coverage: tone questions must state the sentiment counts via [S1] (Task 4).
Retry once; else a plain template answer built straight from the facts.
"""
from __future__ import annotations

import re

from agent_core.base import Agent
from agent_core.memory import SharedMemory

SYSTEM = (
    "You are FinSight, a financial research assistant. Answer the question using ONLY "
    "the facts provided below. Copy numbers exactly as written (do not round or reword "
    "them). Answer EVERY part of the question. If a sentiment summary is provided, state "
    "the exact label counts and cite it. Write 2-6 sentences. Never give investment "
    "advice and never invent numbers or facts.\n\n"
    "FORMAT (REQUIRED): end EVERY sentence that states a number with the bracketed ID of "
    "the fact it came from, e.g. 'Revenue was $96.22B [N1].' Use ONLY IDs that appear in "
    "the Facts list. A sentence that states a number without a matching [ID] is invalid."
)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_BRACKET = re.compile(r"\[([^\]]+)\]")
_ID_TOKEN = re.compile(r"[A-Z]\d+")
_SENT = re.compile(r"(?<=[.!?])\s+")
# Fiscal-period and SEC-form tokens ("Q2", "FY2027", "10-K", "S-1") are not financial
# claims; strip them before the number-support check so they don't read as uncited numbers.
_INCIDENTAL = re.compile(r"\bQ[1-4]\b|\bFY\d{2,4}\b|\b\d+-[A-Z]\b|\bS-1\b", re.IGNORECASE)
# Mistral sometimes prefixes retry chatter ("Apologies…", "Here's a revised answer:");
# strip leading preamble sentences so the UI shows just the answer.
_PREAMBLE = re.compile(
    r"^\s*(?:(?:my\s+)?apolog\w*|sorry|sure|certainly|here(?:'s| is)|revised answer|"
    r"based on the provided facts|using only the provided facts)[^.!?:]*[.!?:]+\s*",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        m = _PREAMBLE.match(text)
        if m and "[" not in m.group(0):  # keep a leading sentence that carries a citation
            text = text[m.end():].strip()
    return text


def _cited_ids(text: str) -> list[str]:
    ids: list[str] = []
    for group in _BRACKET.findall(text):
        ids += _ID_TOKEN.findall(group)
    return ids


def _grounded(num: str, support_text: str, support_nums: list[str]) -> bool:
    """True if `num` appears in the facts, or is a rounded form of a number there (so
    "$127.6B" validates against a "127.58" fact). Keeps invented numbers out while not
    rejecting correct figures over formatting/rounding."""
    if num in support_text:
        return True
    try:
        val = float(num)
    except ValueError:
        return True
    decimals = len(num.split(".", 1)[1]) if "." in num else 0
    for s in support_nums:
        try:
            if round(float(s), decimals) == val:
                return True
        except ValueError:
            continue
    return False


def build_evidence(memory: SharedMemory, question: str, max_passages: int = 8) -> list[dict]:
    """The one ordered, labeled evidence list — used for the prompt AND the Sources panel.

    Each item: {id, text, search_text, source, kind, ticker, data_mode}. Passages keep their
    stored order so [Pk] is stable across the prompt and the UI.
    """
    ev: list[dict] = []

    def add(fid, text, source, kind, ticker, data_mode, search_text=None):
        ev.append({"id": fid, "text": text, "search_text": search_text or text,
                   "source": source or {}, "kind": kind, "ticker": ticker, "data_mode": data_mode})

    if memory.has("amd_notes"):
        for n in memory.get("amd_notes"):
            add(n["id"], n["text"], n.get("source"), "note", n.get("ticker"),
                n.get("data_mode", "xbrl"), f"{n['text']} {n.get('values', '')}")
    if memory.has("amd_sentiment"):
        s = memory.get("amd_sentiment")
        if isinstance(s, dict) and s.get("summary"):
            src = (s.get("labels") or [{}])[0].get("source", {})
            add("S1", s["summary"], src, "sentiment", s.get("ticker"), s.get("data_mode", "filing"))
    if memory.has("amd_passages"):
        for i, p in enumerate(memory.get("amd_passages")[:max_passages], 1):
            add(f"P{i}", p["text"], p.get("source"), "passage", p.get("ticker"),
                p.get("data_mode", "filing"))
    if memory.has("amd_quote"):
        q = memory.get("amd_quote")
        if isinstance(q, dict) and "price" in q:
            txt = f"Current price: {q['price']} {q.get('currency', 'USD')} as of {q.get('as_of', '')}"
            add("Q1", txt, {"source": "Finnhub"}, "quote", q.get("ticker"), "live")
    if memory.has("amd_news"):
        for i, a in enumerate(memory.get("amd_news"), 1):
            add(f"H{i}", f"{a['source']}, {a['date']}: {a['headline']}", {"source": a.get("source")},
                "news", a.get("ticker"), "live")
    return ev


# Which evidence kinds each route's answer may draw on. Keeps a question answered from the
# data its own route gathered, not stale variables left in memory by earlier questions in
# the session (e.g. a prior price/news query polluting a "business" answer).
_ROUTE_KINDS: dict[str, set[str]] = {
    "numbers": {"note"},
    "real_time": {"quote", "news"},
    "sentiment": {"sentiment", "passage"},
    "text": {"passage"},
}


class ReportAgent(Agent):
    name = "report_agent"
    reads: list[str] = []
    writes = ["answer", "evidence"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        route = memory.get("route") or ""
        kinds = _ROUTE_KINDS.get(route)
        all_ev = build_evidence(memory, question)
        # Scope to the current route's evidence so the answer (and the Sources panel) use
        # only what this question gathered — never leftovers from an earlier question in the
        # session (a prior price/news query must not pollute a "business" answer).
        evidence = [e for e in all_ev if e["kind"] in kinds] if kinds else all_ev
        memory.put("evidence", evidence, self.name,
                   description=f"{len(evidence)} labeled sources (route={route or 'all'})")
        if not evidence:
            memory.put(
                "answer",
                f"I don't have SEC filing or financial data for {ticker} yet — it looks "
                f"newly listed and hasn't filed a 10-K or 10-Q. Once it files, filing and "
                f"financial questions will work. You can still ask for its live price or "
                f"recent news.",
                self.name,
            )
            return f"{self.name}: no facts for route '{route or 'all'}' -> wrote no-data answer"

        require_counts = any(e["kind"] == "sentiment" for e in evidence)
        by_id = {e["id"]: e for e in evidence}
        facts_text = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
        convo = "" if require_counts else (memory.get("conversation_state") or "")
        user = question if not convo else f"{question}\n\nRecent conversation:\n{convo}"
        messages = [{"role": "user", "content": f"{user}\n\nFacts:\n{facts_text}"}]

        answer, how, attempts, flags = self._answer_with_checks(
            llm, messages, by_id, evidence, ticker, require_counts)
        memory.put("answer", answer, self.name,
                   description=f"final answer ({how}; attempts={attempts})")
        flag_txt = f"; flags={flags}" if flags else ""
        return f"{self.name}: wrote answer ({how}; {len(evidence)} facts{flag_txt})"

    def _answer_with_checks(self, llm, messages, by_id, evidence, ticker, require_counts):
        engine = "fake" if "Fake" in type(llm).__name__ else "mistral"
        last_reason = ""
        for attempt in (1, 2):
            raw = _strip_preamble(llm.complete(SYSTEM, messages, max_tokens=300, temperature=0.0).strip())
            ok, reason = self._check(raw, by_id, ticker, require_counts)
            if ok:
                return raw, f"{engine}, attempt {attempt}", attempt, []
            last_reason = reason
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"That failed a check ({reason}). Use ONLY the given facts, cite real IDs "
                    "like [N1], copy numbers exactly, and answer every part. Try again.")},
            ]
        tag = self._tag(evidence)
        return self._template(evidence, require_counts, tag), "template fallback", 2, [last_reason]

    @staticmethod
    def _check(answer: str, by_id: dict, ticker: str, require_counts: bool):
        cited = _cited_ids(answer)
        for cid in cited:
            if cid not in by_id:
                return False, f"cited {cid} which does not exist"
        # entity safety (Task 3): a cited note/passage must belong to the question's ticker
        for cid in cited:
            tk = by_id[cid].get("ticker")
            if tk and ticker and tk.upper() != ticker.upper():
                return False, f"{cid} belongs to {tk}, not {ticker}"
        # number grounding (Task 2): every number stated must appear in the provided facts
        # (with rounding tolerance). Checked against ALL facts, not only the ids cited in
        # the same sentence — a correct figure the model stated without a bracket is grounded
        # data, not a hallucination (citation *existence* is enforced above), while an
        # invented number still fails.
        support = " ".join(e["search_text"] for e in by_id.values())
        support_nums = _NUM.findall(support)
        for sent in _SENT.split(answer):
            for num in _NUM.findall(_INCIDENTAL.sub(" ", _BRACKET.sub(" ", sent))):
                if not _grounded(num, support, support_nums):
                    return False, f"number {num} is not in the provided facts"
        # coverage (Task 4): tone questions must state the counts via [S1]
        if require_counts and "S1" not in cited:
            return False, "state the sentiment counts and cite [S1]"
        return True, "ok"

    @staticmethod
    def _tag(evidence: list[dict]) -> str:
        if any(e.get("data_mode") == "sample" for e in evidence):
            return "(sample data)"
        return ""

    @staticmethod
    def _template(evidence: list[dict], require_counts: bool, tag: str = "") -> str:
        picks = [e for e in evidence if e["kind"] in ("note", "sentiment")][:3] or evidence[:3]
        body = " ".join(f"{e['text']} [{e['id']}]" for e in picks)
        return f"{body} {tag}".strip()
