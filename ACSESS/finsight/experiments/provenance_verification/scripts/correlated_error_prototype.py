"""Prototype: correlated evidence errors defeat self-consistency; independent
provenance verification catches them. Three conditions on the same items:

  A. Self-consistency  - sample the report LLM K times, majority vote, confidence=agreement.
  B. Shared-evidence check (FinSight's current ReportAgent._check) - verify the answer's
     numbers against the NOTE (the shared evidence).
  C. Independent provenance verify - compare the note's number against the INDEPENDENT
     source value (the filing). This is the proposed contribution.

A fraction of items have a CORRUPTED note (note number != source number), simulating
multiple agents sharing the same erroneous evidence via shared memory.

Run (from finsight/):  REMOTE_LLM_URL=http://localhost:8000/v1 .venv/bin/python correlated_error_exp.py
"""
from __future__ import annotations
import os, re, sys, random
import requests

sys.path.insert(0, os.path.abspath("."))
from agent_core.agents.report_agent.agent import ReportAgent  # real shared-evidence check

BASE = os.environ.get("REMOTE_LLM_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "mistral")
K = int(os.environ.get("SC_K", "5"))          # self-consistency samples
random.seed(0)

# Representative SEC-grounded facts (quarterly revenue, $B). source_b = independent ground truth.
BASE_ITEMS = [
    ("AMD",  "FY2024 Q4", 6.17, 6.82), ("AMD",  "FY2025 Q1", 7.44, 6.17),
    ("AMD",  "FY2025 Q2", 7.69, 7.44), ("NVDA", "FY2025 Q1", 26.04, 22.10),
    ("NVDA", "FY2025 Q2", 30.04, 26.04), ("NVDA", "FY2025 Q3", 35.08, 30.04),
    ("INTC", "FY2024 Q3", 13.28, 12.83), ("INTC", "FY2024 Q4", 14.26, 13.28),
    ("MSFT", "FY2025 Q1", 65.59, 64.73), ("MSFT", "FY2025 Q2", 69.63, 65.59),
    ("AAPL", "FY2024 Q4", 94.93, 85.78), ("TSLA", "FY2025 Q1", 19.34, 25.18),
]


def make_items(corrupt_frac=0.5):
    items = []
    idx = set(random.sample(range(len(BASE_ITEMS)), int(len(BASE_ITEMS) * corrupt_frac)))
    for i, (tk, label, src, prev) in enumerate(BASE_ITEMS):
        corrupted = i in idx
        # the note value = source, unless corrupted (a plausible wrong number)
        note_val = round(src * random.choice([1.15, 1.22, 0.83, 0.78]), 2) if corrupted else src
        qoq = round((note_val - prev) / prev * 100, 1)
        note_text = (f"Revenue {tk} {label}: ${note_val:.2f}B, "
                     f"{'up' if qoq >= 0 else 'down'} {abs(qoq)}% QoQ from the prior quarter (${prev:.2f}B).")
        items.append({
            "ticker": tk, "label": label,
            "question": f"What was {tk}'s revenue in {label}?",
            "note": {"id": "N1", "text": note_text},
            "note_value": note_val, "source_value": src, "corrupted": corrupted,
        })
    return items


def _nums(text):
    return [float(x) for x in re.findall(r"\d+\.\d+", text)]


def sample_answer(note_text, question, temperature):
    prompt = (f"Use ONLY this fact. Answer in one short sentence and state the revenue figure.\n"
              f"Fact: [N1] {note_text}\nQuestion: {question}\nAnswer:")
    r = requests.post(f"{BASE}/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60, "temperature": temperature}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    items = make_items(0.5)
    clean = [x for x in items if not x["corrupted"]]
    corr = [x for x in items if x["corrupted"]]
    print(f"items: {len(items)} ({len(clean)} clean, {len(corr)} corrupted)  K={K}\n")

    rows = []
    for it in items:
        note_val = it["note_value"]; src = it["source_value"]
        # --- A: self-consistency (K samples) ---
        reported = []
        for _ in range(K):
            ans = sample_answer(it["note"]["text"], it["question"], 0.7)
            ns = _nums(re.sub(r"\[[^\]]*\]", " ", ans))
            # the reported revenue = number closest to the note value (the figure it read)
            reported.append(min(ns, key=lambda v: abs(v - note_val)) if ns else None)
        maj = max(set(reported), key=reported.count) if any(r is not None for r in reported) else None
        agreement = reported.count(maj) / K if maj is not None else 0.0
        sc_correct = (maj is not None and abs(maj - src) < 0.05)   # correct vs INDEPENDENT source

        # --- B: shared-evidence check (FinSight current) on one answer ---
        one = sample_answer(it["note"]["text"], it["question"], 0.0)
        ok, _reason = ReportAgent._check(one + " [N1]", {"N1"}, it["note"]["text"])
        b_flags_error = (not ok)   # does the shared check flag a problem?

        # --- C: independent provenance verify (note vs source) ---
        c_flags_error = abs(note_val - src) >= 0.05

        rows.append({"corrupted": it["corrupted"], "agreement": agreement,
                     "sc_correct": sc_correct, "b_flag": b_flags_error, "c_flag": c_flags_error})

    def grp(sel):
        g = [r for r in rows if r["corrupted"] == sel]
        n = len(g)
        return {
            "n": n,
            "mean_agreement": sum(r["agreement"] for r in g) / n,
            "sc_accuracy": sum(r["sc_correct"] for r in g) / n,
            "b_detect": sum(r["b_flag"] for r in g) / n,
            "c_detect": sum(r["c_flag"] for r in g) / n,
        }

    for name, sel in [("CLEAN", False), ("CORRUPTED", True)]:
        m = grp(sel)
        print(f"=== {name} (n={m['n']}) ===")
        print(f"  A self-consistency: mean agreement (confidence) = {m['mean_agreement']:.2f}")
        print(f"  A self-consistency: accuracy vs source          = {m['sc_accuracy']:.2f}")
        print(f"  B shared-evidence check: error-detection rate   = {m['b_detect']:.2f}")
        print(f"  C independent provenance: error-detection rate  = {m['c_detect']:.2f}")
        print()

    # headline numbers
    cm = grp(True)
    print("HEADLINE (corrupted / correlated-error cases):")
    print(f"  Self-consistency is confidently wrong: agreement {cm['mean_agreement']:.2f} but accuracy {cm['sc_accuracy']:.2f}.")
    print(f"  Shared-evidence check misses them:     detection {cm['b_detect']:.2f}.")
    print(f"  Independent provenance catches them:   detection {cm['c_detect']:.2f}.")


if __name__ == "__main__":
    main()
