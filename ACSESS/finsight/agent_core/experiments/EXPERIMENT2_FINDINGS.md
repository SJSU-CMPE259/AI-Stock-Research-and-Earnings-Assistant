# Experiment 2 — Do we need Python? Is Mistral's bad price the model or missing data?

**Date:** 2026-10-27  **Status:** live run (not simulated)
**Models/data:** Mistral-7B-Instruct on PSC Bridges-2 (Tesla V100, via ollama, OpenAI-compatible :8000); live stock prices + the ground-truth from Finnhub.
**Code:** `agent_core/experiments/exp2_python_vs_llm.py` · **Raw results:** `agent_core/runs/exp2_20261027_125210/` (`calc_results.csv`, `price_results.csv`, `summary.json`)
**Scope:** This file answers the professor's two priority questions only (Python-vs-LLM math, and model-vs-missing-data for prices). The GPT-5/6 and FinBERT comparisons are deliberately left for a later pass.

---

## Experiment A — Python vs Mistral for financial calculations (100 problems)

**Method.** 100 deterministic financial-math problems (QoQ %, YoY %, gross/net margin, EPS, absolute change, 3-quarter average, % of total). Each is computed two ways from the *same* numbers: exact Python vs Mistral doing the arithmetic in its head ("reply with only the number"). A Mistral answer counts as correct if it is within 0.1 absolute **or** 1% relative of the Python value — a generous bar, so only real mistakes are counted wrong.

**Results.**

| | Python | Mistral-7B |
|---|---|---|
| Accuracy (100 problems) | **100%** (exact) | **29%** |
| Gave a usable number | 100% | 100% |
| **→ confidently wrong** | 0% | **71%** |
| Speed per calculation | **0.0007 ms** | 386 ms |
| Relative speed | 1× | **~540,000× slower** |

**Accuracy by calculation type** — it fails hardest on the everyday metrics:

| calc | Mistral accuracy |
|---|---|
| quarter-over-quarter % | **0%** |
| year-over-year % | **0%** |
| gross margin % | **0%** |
| 3-quarter average | 17% |
| net margin % | 23% |
| absolute change ($) | 58% |
| % of total | 67% |
| EPS (division) | 75% |

**Concrete errors (from `calc_results.csv`):**
- QoQ: $25.64B → $23.22B. Python **−9.44%**; Mistral said **+6.83%** — *wrong sign and magnitude.*
- YoY: $23.93B → $16.65B. Python **−30.42%**; Mistral **+31.48%** — *wrong sign.*
- Gross margin: $11.94B / $28.01B. Python **42.63%**; Mistral **23.93%**.
- EPS: $9.79B / 3.13B shares. Python **3.13**; Mistral **0.32** — *it divided the wrong way (shares ÷ income).*

**Conclusion.** Mistral is not a calculator. It never refuses — it emits a confident, wrong number 71% of the time, and it is worst exactly on percentage changes and margins, which are the most common things FinSight reports. Python is exact and ~540,000× faster. **Every number FinSight shows must be computed in Python; the LLM may only phrase the already-computed result.** This is exactly how the pipeline is built today (the `analysis_agent` computes in Python; the `report_agent` LLM is told to copy numbers and is checked for invented ones).

---

## Experiment B — Mistral's wrong price: the model, or missing real-time data? (10 tickers)

**Method.** For 10 large-cap tickers, ask Mistral for the current share price under two conditions: **WITHOUT** any tool (model alone) and **WITH** the live Finnhub quote handed to it as a fact. Compare both to the true live Finnhub price.

**Results.**

| condition | mean error vs real price | within 5% of real |
|---|---|---|
| **WITHOUT** live data (model alone) | **210%** | **0 / 10** |
| **WITH** live data (Finnhub fact) | **0.00%** | **10 / 10** |

**What it made up without data (from `price_results.csv`):**

| ticker | real (Finnhub) | Mistral alone | error |
|---|---|---|---|
| NVDA | $217.95 | $163.00 | 25% |
| AMD | $559.31 | $103.00 | 82% |
| AAPL | $346.96 | $134.70 | 61% |
| TSLA | $407.43 | $1.00 | 100% |
| GOOGL | $322.88 | $1,630.00 | 405% |
| AMZN | $254.21 | $3,400.00 | 1237% |

The GOOGL and AMZN guesses are pre-stock-split prices from its training data — a textbook stale-memory hallucination. **With** the live quote, every answer is exact.

**Conclusion.** The fake price is **a missing-data problem, not a broken model.** Mistral has no real-time connection, so alone it invents plausible-looking (often years-old) numbers. The instant it is given the live Finnhub quote, accuracy goes to 100%. The fix is architectural — a live market-data tool — not a bigger/newer model.

---

## Direct answers to the professor's questions

**Q3 — Can a modern LLM calculate financial values accurately?** No. Mistral-7B scores **29%** on 100 basic financial calcs and is confidently wrong 71% of the time, worst on % changes and margins (0%). Use it for language, not arithmetic.

**Q4 — When do we need Python vs an LLM?** **Python for anything numeric** (percentages, margins, EPS, growth, aggregation) — it is exact and ~540,000× faster. **LLM for language** — turning computed facts into a readable, cited answer. Never let the LLM produce a number that will be displayed.

**Q5 — Is Mistral's error the model or missing real-time data?** **Missing data.** With no tool it is 210% off and 0/10 within 5%; with the live Finnhub quote it is 0.00% off and 10/10. A market-data tool fixes it; a different model would not.

**Q7 (accuracy/affordability/speed, numeric tasks) —** Python wins on all three for calculations: exact, free, sub-millisecond. The LLM's role is the write-up, where its cost/speed are acceptable because it runs once per answer.

**Toward Q6/Q8 (final role of each piece in FinSight):**
- **Python (deterministic tools)** — all calculations and the real-time price/news fetch (Finnhub). Source of every number.
- **Mistral-7B (PSC)** — writes the final cited answer from Python-computed facts; never invents numbers (output is checked).
- **Laya** — routes each question to the right path (best router in Experiment 1).
- *(Open, next pass: GPT-5/6 as a stronger writer/judge; FinBERT vs the current sentiment approach.)*

---

## Limitations
- One model (Mistral-7B); GPT and FinBERT comparisons are future work, as scoped.
- 100 synthetic calc problems with clean inputs — real filings are messier, which can only make LLM arithmetic worse, not better.
- 10 tickers for the price test; the gap (210% vs 0%) is large enough that more tickers would not change the conclusion.
- Numbers are parsed from free text (last number in the reply); the 100% parse rate means this did not distort accuracy.
