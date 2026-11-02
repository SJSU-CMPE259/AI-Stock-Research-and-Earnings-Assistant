# agent_core/eval — comparing decision-makers (rules vs. Mistral vs. Laya)

Harness that runs the **same labeled items** through 5 setups and reports which
decision-maker is best, honestly (with confidence intervals; never a claimed winner
beyond what the data supports).

> **"Jev" in the original spec = Laya** (`convaiinnovations/laya`, a HuggingFace
> text-classification model), used via `transformers`. The paid option is not used.

## Setups (`--setup A|B|C|D|E|all`)
- **A rules** — keyword rules make every decision (no model); Mistral writes answers.
- **B mistral** — Mistral makes every decision; Mistral writes answers.
- **C laya** — Laya makes every decision; Mistral writes answers.
- **D cascade** — Laya first; if its confidence < threshold, ask Mistral and use that.
- **E agree** — ask both; agree → accept; disagree → `needs_review` (sentiment + routing only).

## Decision points (all setups): routing+advice+company, sentiment label+reason,
reranking, answer-check, and full end-to-end.

## Gold data (in this folder) — ⚠️ DRAFT, pending your correction
| File | Rows | Columns |
|------|-----:|---------|
| `routing_questions.csv` | 21 | question, previous_company, gold_intent, gold_company, gold_is_advice |
| `sentiment_statements.csv` | 16 | company, statement, section, gold_label, **gold_reason=TODO**, annotator, double_annotate |
| `rerank_questions.csv` | 10 | question, ticker, gold_section, gold_passage_ids |
| `answers_gold.csv` | 8 | question, ticker, gold_answer_notes, human_supported |
| `e2e_questions.csv` | 24 | question, ticker, note |

Gold rules: **never** use a model's output as gold; `double_annotate=1` rows (~20%) get
a second annotator for Cohen's κ; each set is split 50/50 dev/test with a fixed seed
(in code) — thresholds tuned on dev, results reported on test only.

## Still needed before the live run
1. **Correct the gold labels** above (this is the first STOP).
2. **Sentiment reason taxonomy** (the 9 factors) to fill `gold_reason` — won't invent it.
3. **Laya runtime**: `transformers`+`torch` are not installed. Decide **Mac-CPU install**
   vs **serve Laya on PSC**. Until then the harness uses `FakeLaya`.
4. **Mistral logprobs**: endpoint is Ollama and was offline at draft time; if it returns
   token logprobs we record them as Mistral's confidence, else confidence = "n/a".

## Run (once built)
```bash
python -m agent_core.eval --setup all --split test     # full run
python -m agent_core.eval --dry-run                    # FakeLaya + FakeLLM (CI/offline)
```
Outputs go to `agent_core/runs/eval_<date>/` (`results.csv`, `report.md`, PNG charts).
