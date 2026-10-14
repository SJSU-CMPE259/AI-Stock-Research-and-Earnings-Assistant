"""Classify agent: label each filing passage positive / neutral / negative with Mistral.

Mistral is asked for exactly one word. Anything off-vocabulary is stored as
'uncertain' with needs_review=True. Writes amd_sentiment: per-passage labels, counts
(always with n), and a short summary the report agent can cite as [S1].
"""
from __future__ import annotations

import re

from agent_core.base import Agent
from agent_core.memory import SharedMemory

SYSTEM = (
    "You are a financial sentiment classifier. Reply with exactly ONE word: "
    "positive, neutral, or negative. Output only that word, nothing else."
)
_VOCAB = {"positive", "neutral", "negative"}
_WORD = re.compile(r"[a-z]+")


def _parse_label(reply: str) -> tuple[str, bool]:
    """First word if it is in vocabulary, else ('uncertain', needs_review=True)."""
    words = _WORD.findall((reply or "").lower())
    if words and words[0] in _VOCAB:
        return words[0], False
    return "uncertain", True


class ClassifyAgent(Agent):
    name = "classify_agent"
    reads = ["amd_passages"]
    writes = ["amd_sentiment"]

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        self.check_inputs(memory)
        passages = memory.get("amd_passages")

        labels: list[dict] = []
        counts = {"positive": 0, "neutral": 0, "negative": 0, "uncertain": 0}
        for i, p in enumerate(passages, 1):
            reply = llm.complete(
                SYSTEM,
                [{"role": "user", "content": f"Sentence: {p['text']}\nSentiment (one word):"}],
                max_tokens=5,
                temperature=0.0,
            )
            label, needs_review = _parse_label(reply)
            counts[label] += 1
            labels.append({
                "id": f"L{i}",
                "text": p["text"],
                "label": label,
                "needs_review": needs_review,
                "source": p["source"],
            })

        n = len(passages)
        top = max(("positive", "neutral", "negative"), key=lambda k: counts[k])
        summary = (
            f"{counts[top]} of {n} passages {top} "
            f"({counts['positive']} positive, {counts['neutral']} neutral, "
            f"{counts['negative']} negative"
            + (f", {counts['uncertain']} uncertain/needs review" if counts["uncertain"] else "")
            + ")."
        )

        memory.put(
            "amd_sentiment",
            {"labels": labels, "counts": counts, "n": n, "summary": summary},
            self.name,
            sources=[p["source"] for p in passages],
            description=summary,
        )
        return f"{self.name}: read amd_passages -> wrote amd_sentiment ({counts[top]} of {n} {top})"
