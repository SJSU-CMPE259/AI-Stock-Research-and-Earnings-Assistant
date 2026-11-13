# experiments/provenance_verification

Reproducible experiments for the FinSight paper's central claim: **independently grounded,
provenance-aware verification catches correlated evidence errors that self-consistency and
shared-evidence checking miss.** Full results and exact numbers are in [RESULTS.md](RESULTS.md).

## Layout
```
scripts/   experiment + plotting code
data/      output CSVs (committed so figures/tables are reproducible without re-running)
figures/   rendered figures used in the paper
RESULTS.md consolidated findings, tables, and paper-ready text blocks
```

## Scripts
- `model_setupB_eval.py` — Setup-B (model-only) decision eval on the 100-item routing+sentiment set;
  reasoning-aware (strips `<think>…</think>` for R1). Env: `LLM_MODEL`, `MODEL_MODE` (`direct`/`reason`).
- `mcnemar_mistral_vs_r1.py` — paired McNemar, R1 vs Mistral, on the same 100 items.
- `correlated_error_prototype.py` — synthetic single-note corruption demo (3 conditions).
- `corruption_sweep.py` — corruption-rate sweep (synthetic); `MODEL_MODE=reason` for R1.
- `finrank_value_corruption.py` — **primary experiment**: corrupt one real figure in a FinRank gold
  passage; self-consistency vs shared-check (B) vs independent provenance (C); bootstrap CIs + McNemar.
- `finrank_hardneg_retrieval.py` — secondary: hard-negative (retrieval) corruption → abstention contrast.
- `plot_*.py` — render the figures from the CSVs in `data/`.

## Running
Requires a running model endpoint (ollama OpenAI-compatible) reachable at `REMOTE_LLM_URL`
(default `http://localhost:8000/v1`) and, for routing, a Laya router at `LAYA_URL`
(`http://localhost:8001`). We served Mistral-7B, Mistral-NeMo-12B, and DeepSeek-R1-8B on a single
NVIDIA A100-SXM4-40GB (Jetstream2 `g3.xl`).

```bash
# from finsight/ (so agent_core is importable)
REMOTE_LLM_URL=http://localhost:8000/v1 LLM_MODEL=mistral \
  python experiments/provenance_verification/scripts/finrank_value_corruption.py
REMOTE_LLM_URL=http://localhost:8000/v1 LLM_MODEL=deepseek-r1:8b MODEL_MODE=reason N_ITEMS=40 \
  python experiments/provenance_verification/scripts/finrank_value_corruption.py
```

## FinRank data (not committed)
The FinRank benchmark is external. Clone it next to this folder before running the FinRank scripts
(the scripts expect `scripts/FinRank/FinRank.jsonl`, or adjust the `DATA` path):
```bash
git clone https://github.com/datanxt/FinRank.git
```
We do not redistribute FinRank here; see its repository and paper (arXiv 2608.07400) for license/terms.
