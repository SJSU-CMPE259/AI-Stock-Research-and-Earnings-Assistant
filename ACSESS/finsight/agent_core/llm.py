"""MistralLLM (wraps the existing backend get_llm()) + FakeLLM for offline runs.

MistralLLM does NOT edit backend/app/llm.py — it calls get_llm().complete(). If the
remote endpoint rejects a 'system' role, it retries once with the instructions folded
into the first user message.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class MistralLLM:
    """Mistral-7B on PSC via backend.app.llm.get_llm().complete()."""

    def __init__(self) -> None:
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from backend.app.llm import get_llm  # lazy: only needed in real mode

            self._client = get_llm()
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        client = self._client_lazy()
        try:
            return client.complete(system, messages, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"system-role call failed ({e}); folding system into user message")
            return client.complete("", _fold_system(system, messages), max_tokens, temperature)


def _fold_system(system: str, messages: list[dict]) -> list[dict]:
    """Put the system instructions at the top of the first user message instead."""
    folded = list(messages)
    if folded and folded[0].get("role") == "user":
        folded[0] = {"role": "user", "content": f"{system}\n\n{folded[0]['content']}"}
    else:
        folded.insert(0, {"role": "user", "content": system})
    return folded


class FakeLLM:
    """Offline stand-in with the same call surface. Deterministic, no network.

    Pass `responses` to script exact replies; otherwise it returns a canned answer
    shaped by the prompt (used by the demo and tests).
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"system": system, "messages": messages})
        if self.responses:
            return self.responses.pop(0)
        blob = system + " " + " ".join(m.get("content", "") for m in messages)
        low = blob.lower()
        if "one word" in low and "positive" in low and "negative" in low:
            return "positive"
        if "facts:" in low or "cite" in low:
            m = re.search(r"\[([A-Z]\d+)\]", blob)
            cite = f" [{m.group(1)}]" if m else ""
            return f"Here is a brief summary based on the provided facts{cite}. (sample data)"
        return "(fake response)"
