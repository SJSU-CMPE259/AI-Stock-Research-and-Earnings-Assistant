"""Report agent: write the final answer with Mistral, cite fact IDs, verify, fall back.

Rules given to Mistral: use ONLY the provided facts, copy numbers exactly, cite IDs
like [N1], 2-5 sentences, no advice. Then we CHECK the answer: every cited ID must
exist and every number must appear in the facts. Retry once; if it still fails, build
a plain template answer straight from the facts so the demo never shows a made-up
number.
"""
from __future__ import annotations

import re

from agent_core.base import Agent
from agent_core.memory import SharedMemory

SYSTEM = (
    "You are FinSight, a financial research assistant. Answer the question using ONLY "
    "the facts provided below. Copy numbers exactly as written. Cite the facts you use "
    "by their IDs in square brackets, like [N1]. Write 2-5 sentences. Never give "
    "investment advice and never invent numbers or facts."
)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_BRACKET = re.compile(r"\[([^\]]+)\]")
_ID_TOKEN = re.compile(r"[A-Z]\d+")


def _cited_ids(answer: str) -> list[str]:
    """IDs cited inside brackets, including combined forms like [N1, N2, N3]."""
    ids: list[str] = []
    for group in _BRACKET.findall(answer):
        ids += _ID_TOKEN.findall(group)
    return ids


def _select_passages(passages: list[dict], question: str, k: int = 3) -> list[dict]:
    """Pick the k passages with the most word overlap with the question (no vector DB)."""
    qwords = {w.strip(".,?'\"").lower() for w in question.split() if len(w) > 2}
    scored = [
        (len(qwords & {w.strip(".,?'\"").lower() for w in p["text"].split()}), i, p)
        for i, p in enumerate(passages)
    ]
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [p for _, _, p in scored[:k]]


class ReportAgent(Agent):
    name = "report_agent"
    reads: list[str] = []
    writes = ["answer"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        facts, facts_text, valid_ids, facts_blob = self._gather(memory, question)
        if not facts:
            memory.put("answer", "I don't have the data to answer that yet. (sample data)", self.name)
            return f"{self.name}: no facts in memory -> wrote fallback answer"

        convo = memory.get("conversation_state") or ""
        user = question if not convo else f"{question}\n\nRecent conversation:\n{convo}"
        messages = [{"role": "user", "content": f"{user}\n\nFacts:\n{facts_text}"}]

        tag = self._source_tag(memory)
        answer, how = self._answer_with_checks(llm, messages, valid_ids, facts_blob, facts, tag)
        memory.put("answer", answer, self.name, description=f"final answer ({how})")
        return f"{self.name}: wrote answer ({how}; {len(valid_ids)} facts available)"

    def _gather(self, memory: SharedMemory, question: str):
        facts: list[tuple[str, str]] = []
        lines: list[str] = []
        blob: list[str] = []

        if memory.has("amd_notes"):
            for n in memory.get("amd_notes"):
                facts.append((n["id"], n["text"]))
                lines.append(f"[{n['id']}] {n['text']}")
                blob.append(f"{n['text']} {n.get('values', '')}")
        if memory.has("amd_sentiment"):
            s = memory.get("amd_sentiment")
            if isinstance(s, dict) and s.get("summary"):
                facts.append(("S1", s["summary"]))
                lines.append(f"[S1] {s['summary']}")
                blob.append(s["summary"])
        if memory.has("amd_passages"):
            for i, p in enumerate(_select_passages(memory.get("amd_passages"), question), 1):
                facts.append((f"P{i}", p["text"]))
                lines.append(f"[P{i}] {p['text']}")
                blob.append(p["text"])
        if memory.has("amd_quote"):
            q = memory.get("amd_quote")
            if isinstance(q, dict) and "price" in q:
                txt = f"Current price: {q['price']} {q.get('currency', 'USD')} as of {q.get('as_of', '')}"
                facts.append(("Q1", txt))
                lines.append(f"[Q1] {txt}")
                blob.append(txt)
        if memory.has("amd_news"):
            for i, a in enumerate(memory.get("amd_news"), 1):
                txt = f"{a['source']}, {a['date']}: {a['headline']}"
                facts.append((f"H{i}", txt))
                lines.append(f"[H{i}] {txt}")
                blob.append(txt)

        valid_ids = {fid for fid, _ in facts}
        return facts, "\n".join(lines), valid_ids, " ".join(str(b) for b in blob)

    @staticmethod
    def _source_tag(memory: SharedMemory) -> str:
        """Label the data provenance: live Finnhub quote/news vs sample fixtures."""
        q = memory.get("amd_quote") if memory.has("amd_quote") else None
        if (isinstance(q, dict) and q.get("live")) or memory.has("amd_news"):
            return "(live: Finnhub)"
        return "(sample data)"

    def _answer_with_checks(self, llm, messages, valid_ids, facts_blob, facts, tag="(sample data)"):
        for attempt in (1, 2):
            raw = llm.complete(SYSTEM, messages, max_tokens=300, temperature=0.0).strip()
            ok, reason = self._check(raw, valid_ids, facts_blob)
            if ok:
                return raw, f"mistral, attempt {attempt}"
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"That failed a check ({reason}). Use ONLY the given facts, cite real "
                    "IDs like [N1], and copy numbers exactly. Try again."
                )},
            ]
        return self._template(facts, tag), "template fallback"

    @staticmethod
    def _check(answer: str, valid_ids: set[str], facts_blob: str):
        for cid in _cited_ids(answer):
            if cid not in valid_ids:
                return False, f"cited {cid} which does not exist"
        answer_no_cites = _BRACKET.sub(" ", answer)  # don't read ID digits as numbers
        blob_nums = set(_NUM.findall(facts_blob))
        for num in _NUM.findall(answer_no_cites):
            if num not in blob_nums and num not in facts_blob:
                return False, f"number {num} not in the facts"
        return True, "ok"

    @staticmethod
    def _template(facts: list[tuple[str, str]], tag: str = "(sample data)") -> str:
        body = " ".join(f"{txt} [{fid}]" for fid, txt in facts[:3])
        return f"{body} {tag}"
