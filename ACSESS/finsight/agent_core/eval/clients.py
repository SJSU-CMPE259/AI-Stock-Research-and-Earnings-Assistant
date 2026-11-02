"""Decision-maker clients, one interface: decide_choice() and decide_yesno().

- RulesDecider  : keyword rules (setup A), no model.
- MistralDecider: Mistral via the Ollama OpenAI endpoint, confidence from logprobs.
- LayaDecider   : Laya Router served over HTTP (laya[serve] or our wrapper).
- FakeLLM / FakeLaya : deterministic stand-ins for --dry-run and CI (no network).

Every call returns a Decision so the harness records the same fields for all setups.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class Decision:
    answer: str | None          # the parsed decision (an option, or "yes"/"no"), or None
    confidence: float           # 0..1 ("n/a" cases use 0.0 and valid=False upstream)
    valid: bool                 # did the raw output parse to an allowed answer?
    latency_s: float = 0.0
    source: str = ""            # "rules" | "mistral" | "laya" | "fake"
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None


def _first_word(text: str) -> str:
    m = re.search(r"[a-zA-Z_]+", text or "")
    return m.group(0).lower() if m else ""


# ---- A: rules -------------------------------------------------------------

class RulesDecider:
    source = "rules"

    def decide_choice(self, state: str, instructions: str, options: dict[str, str]) -> Decision:
        """Keyword routing/sentiment rules. `options` maps option -> keyword hints."""
        low = state.lower()
        for opt, hints in options.items():
            if any(h.strip() and h.strip() in low for h in str(hints).split(",")):
                return Decision(opt, 1.0, True, source=self.source)
        # default to the last option (e.g. "text"/"neutral") — rules are deterministic
        return Decision(list(options)[-1], 0.5, True, source=self.source)

    def decide_yesno(self, state: str, instructions: str, keywords: list[str]) -> Decision:
        low = state.lower()
        yes = any(k in low for k in keywords)
        return Decision("yes" if yes else "no", 1.0, True, source=self.source)


# ---- B: Mistral (Ollama endpoint, with logprobs) --------------------------

class MistralDecider:
    source = "mistral"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base = (base_url or os.environ.get("REMOTE_LLM_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "mistral")

    def _ask(self, prompt: str) -> tuple[str, float, int, int, float]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5, "temperature": 0, "logprobs": True, "top_logprobs": 1,
        }
        start = time.time()
        r = requests.post(f"{self.base}/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        text = ch["message"]["content"]
        conf = 0.0
        lp = (ch.get("logprobs") or {}).get("content") or []
        if lp:
            conf = math.exp(lp[0]["logprob"])  # prob of the first generated token
        usage = d.get("usage", {})
        return text, conf, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), time.time() - start

    def decide_choice(self, state: str, instructions: str, options: dict[str, str]) -> Decision:
        opts = list(options)
        prompt = (f"{instructions}\nAnswer with exactly one word from: {', '.join(opts)}.\n"
                  f"Text: {state}\nAnswer:")
        text, conf, tin, tout, lat = self._ask(prompt)
        word = _first_word(text)
        valid = word in opts
        return Decision(word if valid else None, conf, valid, lat, self.source, tin, tout, text)

    def decide_yesno(self, state: str, instructions: str, keywords=None) -> Decision:
        prompt = f"{instructions}\nAnswer with exactly one word, yes or no.\nText: {state}\nAnswer:"
        text, conf, tin, tout, lat = self._ask(prompt)
        word = _first_word(text)
        valid = word in ("yes", "no")
        return Decision(word if valid else None, conf, valid, lat, self.source, tin, tout, text)


# ---- C: Laya (served over HTTP) -------------------------------------------

class LayaDecider:
    source = "laya"

    def __init__(self, base_url: str | None = None):
        self.base = (base_url or os.environ.get("LAYA_URL", "http://localhost:8001")).rstrip("/")

    def _predict(self, state: str, questions: dict) -> tuple[dict, int, float]:
        start = time.time()
        r = requests.post(f"{self.base}/predict", json={"state": state, "questions": questions}, timeout=60)
        r.raise_for_status()
        d = r.json()
        return d, d.get("usage", {}).get("input_tokens", 0), time.time() - start

    def decide_choice(self, state: str, instructions: str, options: dict[str, str]) -> Decision:
        q = {"q": {"type": "choice", "instructions": instructions, "criteria": dict(options)}}
        d, tin, lat = self._predict(state, q)
        a = d["answers"]["q"]
        return Decision(a.get("choice"), float(a.get("answer_confidence", a.get("confidence", 0.0))),
                        a.get("choice") in options, lat, self.source, tin, 0, a)

    def decide_yesno(self, state: str, instructions: str, keywords=None) -> Decision:
        q = {"q": {"type": "noul", "instructions": instructions}}
        d, tin, lat = self._predict(state, q)
        a = d["answers"]["q"]
        noul = float(a.get("noul", 0.0))
        return Decision("yes" if noul >= 0.5 else "no",
                        float(a.get("answer_confidence", a.get("confidence", 0.0))),
                        True, lat, self.source, tin, 0, a)


# ---- fakes (offline / CI) -------------------------------------------------

class FakeLLM:
    """Deterministic Mistral stand-in: applies the same keyword rules, fixed confidence."""
    source = "fake-mistral"

    def __init__(self):
        self._rules = RulesDecider()

    def decide_choice(self, state, instructions, options):
        d = self._rules.decide_choice(state, instructions, options)
        return Decision(d.answer, 0.8, True, 0.001, self.source, 20, 1)

    def decide_yesno(self, state, instructions, keywords=None):
        d = self._rules.decide_yesno(state, instructions, keywords or [])
        return Decision(d.answer, 0.8, True, 0.001, self.source, 20, 1)


class FakeLaya:
    """Deterministic Laya stand-in with the same call surface + a confidence."""
    source = "fake-laya"

    def __init__(self):
        self._rules = RulesDecider()

    def decide_choice(self, state, instructions, options):
        d = self._rules.decide_choice(state, instructions, options)
        return Decision(d.answer, 0.9, True, 0.001, self.source, 15, 0)

    def decide_yesno(self, state, instructions, keywords=None):
        d = self._rules.decide_yesno(state, instructions, keywords or [])
        return Decision(d.answer, 0.9, True, 0.001, self.source, 15, 0)
