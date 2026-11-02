# Experiment 4 — A newer GPT model (gpt-5) vs Python on financial math

**Date:** 2026-10-27  **Status:** live run (paid OpenAI API)
**Models/data:** OpenAI **gpt-5** (reasoning_effort = minimal), compared to exact Python and to Mistral-7B's numbers from Experiment 2 — the **same 50 calc problems** (same seed).
**Code:** `agent_core/experiments/exp4_gpt_vs_python.py` · **Raw results:** `agent_core/runs/exp4_20261027_140623/` (`gpt_calc_results.csv`, `summary.json`)
**Why:** the professor asked to test Mistral against a newer GPT model (GPT-5/6) and to check whether Mistral-7B is still a strong, current choice.
**Cost:** the 50-question run used **3,270 input + 651 output = 3,921 tokens** — about a cent. Kept cheap on purpose: 50 questions, one-number answers, minimal reasoning effort.

---

## Method

The identical 50 problems from Experiment 2 (QoQ %, YoY %, gross/net margin, EPS, absolute change, 3-quarter average, % of total), each with an exact Python ground truth. gpt-5 is asked to compute each ("reply with only the number"); a reply is correct within 0.1 absolute or 1% relative — the same generous bar used for Mistral.

**gpt-5 setup note:** gpt-5 rejects `temperature=0` (only the default is allowed) and shares its token budget between hidden reasoning and the answer, so a tiny cap returns an *empty* reply. We use `reasoning_effort="minimal"` with room to answer — fast, cheap, and enough for arithmetic.

---

## Results — three ways to compute the same 50 problems

| | Python | **gpt-5** | Mistral-7B |
|---|---|---|---|
| Accuracy | **100%** (exact) | **98%** (49/50) | **29%** |
| Gave a usable number | 100% | 100% | 100% |
| Speed per calc | **0.0007 ms** | 1,323 ms | 386 ms |
| Cost per 50 | free | ~1 cent | free (self-hosted) |

**gpt-5 accuracy by calculation type:**

| calc | gpt-5 | (Mistral) |
|---|---|---|
| quarter-over-quarter % | 100% | 0% |
| year-over-year % | 100% | 0% |
| gross margin % | 100% | 0% |
| net margin % | 100% | 23% |
| EPS | 100% | 75% |
| absolute change | 100% | 58% |
| % of total | 100% | 67% |
| 3-quarter average | **83%** | 17% |

**gpt-5's only miss:** a 3-quarter average — Python **24.05**, gpt-5 **24.71** (it averaged slightly wrong). One error in 50.

---

## What this says

**1. A newer model is dramatically better than Mistral-7B at financial math** — **98% vs 29%** on the same questions. Where Mistral scored 0% (percentage changes, gross margin), gpt-5 is perfect. So on the professor's question "is Mistral-7B still a strong, current model?": **for arithmetic, no** — a 2026 model is far better, and Mistral-7B's calculation weakness is a real limitation, not a quirk of our prompts.

**2. But this does NOT change the "use Python for the numbers" conclusion.** Even gpt-5:
- is **not perfect** (missed a simple average; at scale, some wrong numbers will slip through),
- is **~1.9 million× slower** than Python (1,323 ms vs 0.0007 ms) and ~3× slower than local Mistral,
- **costs money per call**, where Python is free and exact.

For values FinSight *displays*, Python remains the right engine: exact, instant, free, auditable. A bigger LLM narrows the gap but never beats "compute it directly."

**3. Where a newer model earns its place is language, not arithmetic** — writing the answer, handling messy phrasing, judging/comparing. That is the open follow-up (gpt-5 as a stronger *writer/judge* than Mistral), separate from the calculation path.

---

## Direct answers to the professor's questions

- **Q3 (can a modern LLM calculate accurately?)** — Much better than before: gpt-5 hit **98%** vs Mistral's 29%. But "much better" ≠ exact or free, so it still shouldn't be the source of displayed numbers.
- **Q4 (when Python vs LLM?)** — Unchanged: **Python for every number** (exact, free, instant); **LLM for language**. gpt-5 reinforces this — the gap to Mistral is about *language model quality*, while Python is in a different category for arithmetic.
- **Q6 (which performs best?)** — For calculation: **Python > gpt-5 (98%) > Mistral (29%)**. For *writing* the answer: a newer model like gpt-5 is the upgrade candidate over Mistral (future test).
- **Q7 (accurate / affordable / fast?)** — For numbers, **Python wins all three**. Between the two LLMs, gpt-5 is far more accurate but costs money and is slower; Mistral is free (self-hosted on PSC) but weak at math.

---

## Limitations
- One newer model (gpt-5) at minimal reasoning effort; higher effort might fix the single average miss but costs more and is slower — not worth it when Python is exact.
- 50 problems (kept small to limit spend); clean synthetic inputs — real filings are messier.
- gpt-5 pricing not asserted exactly here; the token count (3,921) is reported so the spend is auditable.
- Mistral numbers are from Experiment 2 (same 50 problems, same seed), not re-run here.
