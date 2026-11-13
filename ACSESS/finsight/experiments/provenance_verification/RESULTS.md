# Provenance-grounded verification vs. correlated evidence errors

Research experiments supporting the FinSight paper's central claim:

> **Self-consistency produces false confidence when agents share the same erroneous evidence;
> independently grounded, provenance-aware verification reduces these correlated errors.**

All numbers below are measured. Models served via ollama (OpenAI-compatible endpoint, 4-bit
quantized) on a single **NVIDIA A100-SXM4-40GB** (Jetstream2 `g3.xl`).

---

## Experiment 1 — Model comparison on the decision set (routing + sentiment)

Setup B (model-only decision-maker) on 60 routing + 40 sentiment = **100 classification items**.
Reproduced the repo harness to 4 decimals; `data/` and `scripts/model_setupB_eval.py`.

| Metric | Mistral-7B | Mistral-NeMo-12B | DeepSeek-R1-8B |
|---|---|---|---|
| Accuracy (intent+sentiment), 95% CI | 0.6800 [0.60, 0.77] | 0.68 [0.60, 0.77] | 0.7700 [0.69, 0.85] |
| Intent accuracy | 0.5167 | — | 0.6667 |
| Intent macro-F1 | 0.3683 | 0.35 | 0.4784 |
| Sentiment accuracy | 0.9250 | 0.90 | 0.9250 |
| Sentiment macro-F1 | 0.9212 | — | 0.9244 |
| Advice recall | 0.5000 | 0.38 | 0.5000 |
| False-refusal rate | 0.0769 | 0.00 | 0.0192 |
| Valid-output rate | 0.9700 | 0.99 | 1.0000 |
| Mean latency / decision | 262 ms | 305 ms | 2588 ms |

**Paired McNemar (DeepSeek-R1-8B vs Mistral-7B)** on the 100 items (one matched run,
R1 = 79/100, Mistral = 68/100): discordant 15 (R1 right / Mistral wrong) vs 4 (reverse);
exact two-sided **p = 0.0192**.

**Findings.** Scaling Mistral 7B → 12B gives no gain (0.68 → 0.68). A *reasoning* model
(R1-8B, smaller on disk than NeMo-12B) is the only one that helps, concentrated on the weak
routing-intent task (F1 0.3683 → 0.4784), at ~10× latency. The gain is **consistent with**
reasoning helping, not scale. R1 runs at temperature 0.6 (its recommended setting) and is
therefore stochastic: two runs scored 0.77 and 0.79; the McNemar is reported on the matched 0.79 run.

---

## Experiment 2 — Correlated-error prototype (synthetic)

Controlled demonstration of the mechanism: a note (shared evidence) whose numeric value is
corrupted for a fraction of items. Three conditions: (A) self-consistency, (B) the repo's current
shared-evidence check (`ReportAgent._check`), (C) independent provenance (note value vs. source).
20 items, K=5. `scripts/corruption_sweep.py`, `data/corruption_sweep_*.csv`,
`figures/correlated_error_figure.png`, `figures/reasoning_vs_provenance.png`.

Corruption-rate sweep (identical for Mistral-7B and DeepSeek-R1-8B):

| p (corrupt) | SC confidence | SC accuracy | silent err: no verify | silent err: B (current) | silent err: C (provenance) |
|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| 0.25 | 1.00 | 0.75 | 0.25 | 0.25 | 0.00 |
| 0.50 | 1.00 | 0.50 | 0.50 | 0.50 | 0.00 |
| 0.75 | 1.00 | 0.25 | 0.75 | 0.75 | 0.00 |
| 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |

Self-consistency confidence stays 1.00 while accuracy falls to 0 (false confidence). The current
shared-evidence check sits exactly on the no-verification line (it validates the answer against the
*corrupted* note, so it passes). Independent provenance holds silent error at 0. Both models behave
identically here — reasoning does not resist the corruption. (These exact 1.00/0.00 values reflect
the synthetic/oracle setup; Experiment 3 is the realistic, benchmark-grounded version.)

---

## Experiment 3 — FinRank value corruption (PRIMARY)

Benchmark: **FinRank** (Mansouri et al., arXiv 2608.07400; github.com/datanxt/FinRank) — 1,185
analyst-authored QA records over 10-K/10-Q of 22 companies (pharma, oil & gas, automotive),
with gold passages and hard negatives. We use the subset whose reference-answer figure is
verbatim-present in a gold passage (413 eligible).

**Correlated-error model (value corruption):** select a TARGET figure in both the reference answer
and a gold passage; feed the model either the clean gold passage or the same passage with that one
figure replaced by a plausible wrong value (×{1.18, 1.27, 0.82, 0.73}). Conditions as in Exp. 2,
but C verifies against the *original, uncorrupted* gold. `scripts/finrank_value_corruption.py`,
`data/finrank_vc_*.csv`, `figures/finrank_mistral_figure.png`, `figures/finrank_overlay.png`.

**Design note / secondary result.** FinRank's *hard negatives* (irrelevant passages) instead make
the model **abstain** rather than confabulate, so retrieval-swap does not reproduce the failure
(B and C collapse together). The dangerous vector is value corruption within a plausible passage.
`scripts/finrank_hardneg_retrieval.py` records this contrast.

### Results

**Mistral-7B (n = 60 corrupted items)**
- Clean: self-consistency agreement 0.55, accuracy 0.50. Corrupted: agreement 0.59, **accuracy 0.00**.
- Detection on corrupted evidence: **B (shared-check) 0.35** [0.23, 0.47]; **C (provenance) 0.77** [0.65, 0.87].
- Paired McNemar (C vs B): 25 (C caught / B missed) vs 0 (reverse); exact **p = 5.96×10⁻⁸**.
- Silent-error vs corrupted fraction: 0.50→1.00 (no verify), 0.50→0.65 (B), 0.50→0.23 (C).

**DeepSeek-R1-8B (n = 40 corrupted items)**
- Clean: agreement 0.26, accuracy 0.35. Corrupted: agreement 0.20, **accuracy 0.03**.
- Detection on corrupted evidence: **B 0.68** [0.53, 0.80]; **C 0.90** [0.80, 0.97].
- Paired McNemar (C vs B): 9 vs 0; **p = 0.003906**.

**Findings.**
1. On real SEC-filing evidence, self-consistency gives false confidence under value corruption:
   confidence stays flat (~0.55–0.59 Mistral) while accuracy collapses to 0.
2. Verifying against the shared/retrieved evidence is insufficient (Mistral detects 35%).
3. Independent provenance verification detects 77% (Mistral) / 90% (R1) and **strictly dominates**
   the shared check for both models (one-directional discordance; McNemar p = 5.96×10⁻⁸ / 0.0039),
   cutting Mistral silent error from 0.65 to 0.23 at full corruption.
4. Reasoning does not rescue: R1's accuracy also collapses (to 0.03). R1 is noisier (lower clean
   accuracy and sample agreement) so the two models are not identical, but the core claim holds for both.

### Limitations
- Clean-context accuracy (0.50 Mistral / 0.35 R1) is a noise floor from extracting figures out of raw
  tabular passages; it affects all conditions equally and sets the p=0 silent-error baseline.
- n = 60 / 40 from the verbatim-extractable subset; single corruption-magnitude band; 4-bit models.
- Detection/accuracy use significant-figure numeric grounding, not full answer semantics.
- Single run; self-consistency uses temperature 0.7 (R1 uses 0.6, so R1 is stochastic across runs).
