# Experiment 5 — Current open models (newest Mistral, DeepSeek R1) vs Python on financial math

**Date:** 2026-11-02  **Status:** live run (free, Google Colab T4 GPU, Ollama)
**Models/data:** **Ministral 3 8B** (`ministral-3:8b`, newest small Mistral family) and **DeepSeek R1 8B** (`deepseek-r1:8b`, open reasoning model, Qwen-distill), on the **same 100 calc problems** as Experiment 2 (same generator, seed 0, same prompt `CALC_SYS`, same correctness bar).
**Code:** `agent_core/experiments/exp5_open_models.py` · Colab notebook used to run it: install Ollama → pull models → run the script per model.
**Raw results:** the run folders (`calc_results.csv`, `summary.json`) were lost when the Colab VM was recycled after the ~5 h R1 run. The full console output survives and is saved in `agent_core/experiments/exp5_logs/` (every per-question line plus the summary).
**Why:** the professor asked whether Mistral-7B is still a current model, and whether a newer model changes the "Python for numbers" decision. Exp 4 answered that with a paid model (gpt-5); this answers it with **free, open** models we could self-host.
**Cost:** free (Colab GPU).

---

## Method

Same as Experiments 2 and 4: each problem has an exact Python answer; the model is asked to "reply with only the final number"; a reply is correct within 0.1 absolute or 1% relative. Ollama `/api/chat`, temperature 0 for Ministral, 0.6 for R1 (DeepSeek's recommended setting), output cap 4,096 tokens. For R1 the reasoning trace is stripped and only the final answer is scored.

---

## Results

| | Python | gpt-5 (Exp 4) | **Ministral 3 8B** | Mistral-7B (Exp 2) | **DeepSeek R1 8B** |
|---|---|---|---|---|---|
| Accuracy, all 100 | **100%** | — | **72%** | 29% | did not finish |
| Accuracy, first 50 (gpt-5's set) | **100%** | **98%** | **74%** | — | did not finish |
| Gave a usable number | 100% | 100% | 100% | 100% | 39% (fragments) |
| Speed per calc | 0.0007 ms | 1,323 ms | **~300 ms** (602 ms mean incl. model load) | 386 ms | **~175,000 ms (~3 min)** |
| Cost | free | paid | free | free | free |

**Ministral 3 8B by calculation type (100 problems):**

| calc | Ministral 3 8B |
|---|---|
| gross margin % | 100% |
| net margin % | 100% |
| EPS | 100% |
| % of total | 92% |
| absolute change | 58% |
| year-over-year % | 46% |
| 3-quarter average | 42% |
| quarter-over-quarter % | 38% |

**The sign error.** All 5 of Ministral's `abs_change` misses are the right magnitude with the **minus sign dropped** (e.g. Python −11.30, Ministral 11.30; −2.26 → 2.26; −0.35 → 0.35). In a finance answer that turns a drop into a gain — and it looks plausible, so a reader would not catch it.

**DeepSeek R1 8B.** Every one of the 100 calls ran to the 4,096-token cap while still reasoning (~170–200 s each), so it never produced a final answer. The few numbers parsed (3.0, 20.0, 1.0, …) are fragments of truncated output. **This is a completion/latency result, not an accuracy score** — R1's arithmetic ability was not measured.

---

## What this says

**1. The newest Mistral is a real upgrade over Mistral-7B: 72% vs 29%.** One-step ratios (margins, EPS) are now perfect. So on "is Mistral-7B still current?" — no; Ministral 3 is the better Mistral today, at the same size and still free.

**2. It still is not safe as the source of displayed numbers.** ~1 in 4 answers is wrong, multi-step calculations (QoQ, YoY, averages) are below 50%, and the errors are plausible-looking (dropped signs). Python remains exact, instant and free.

**3. A reasoning model is impractical here.** R1 8B took ~3 minutes per calculation on a T4 and did not finish within 4k tokens. Python answers in under a millisecond. For an interactive assistant that is disqualifying regardless of eventual accuracy.

**4. Ranking for calculation:** Python (100%) > gpt-5 (98%, paid) > Ministral 3 8B (72%, free) > Mistral-7B (29%) ; R1 8B did not complete.

---

## Direct answers to the professor's questions

- **Q3 (can an LLM calculate financial values accurately?)** — The best open small model gets 72%; gpt-5 gets 98%. Neither is exact; Python is.
- **Q4 (when Python vs LLM?)** — Unchanged: Python for every number, the LLM for language.
- **Q6/Q7 (best / most accurate, affordable, fast?)** — For numbers Python wins on all three. If we upgrade the *writer* model, Ministral 3 8B is the free, same-size candidate to replace Mistral-7B.
- **Professor's point 4 (did we try the latest Mistral?)** — Yes: Ministral 3 8B, this experiment.

---

## Limitations
- One run per model; no repeats, so no variance estimate.
- Different hardware/serving from Exp 2 (Colab T4 + Ollama 4-bit quantized vs PSC). Mistral-7B was not re-scored on the identical Colab setup in this write-up, so part of the 72% vs 29% gap could come from the setup, not only the model.
- R1 was tried with one configuration (4,096-token cap, Ollama's default context window, our "no working" system prompt — DeepSeek recommends no system prompt). A larger cap/context might let it finish, but at ~3 min per 4k tokens it would still be far too slow.
- Raw per-row CSVs were lost; results come from the saved console logs.
- Clean synthetic inputs; real filings are messier.
