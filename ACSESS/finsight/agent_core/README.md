# FinSight agent core

A small multi-agent core for FinSight. Agents never call each other — they read and
write named items in one **shared memory**. An **orchestrator** routes each question
to the right agents using simple Python rules, runs them in order, and returns a trace.

> Self-contained: nothing outside `agent_core/` is edited. The LLM is Mistral-7B on PSC,
> reached through the existing `backend/app/llm.py`; a **FakeLLM** lets everything run
> offline. All answers built from the bundled sample data end with `(sample data)`.

## Run the demo

From `ACSESS/finsight/`:

```bash
python -m agent_core.run_demo --fake   # offline (FakeLLM, no GPU)
python -m agent_core.run_demo          # real Mistral (needs the PSC tunnel on :8000)
```

It asks 5 questions in one conversation and prints, for each: the question → route →
each agent (what it read / wrote) → the answer → time. Traces are saved under `runs/`
(gitignored).

## Pieces

| File | What it does |
|------|--------------|
| `memory.py` | `SharedMemory` (put/get/has/list/describe/save) + `Conversation` (last 3 Q/A pairs) |
| `base.py` | `Agent` base class (`reads`, `writes`, `run()`) |
| `llm.py` | `MistralLLM` (wraps `get_llm()`) + `FakeLLM` (offline) |
| `orchestrator.py` | keyword router + run order + trace |
| `agents/<name>/agent.py` | the 5 agents: data, analysis, classify, report, market |
| `data/sample_amd.json` | made-up sample data (fiscal labels; `"sample": true`) |
| `run_demo.py` | the 5-question demo |

## Routing (keyword rules, in order)

1. buy / sell / hold / "should I invest" → **refuse** (no agents, no LLM)
2. price / today / "right now" / current → **real_time**: market → report
3. revenue / income / EPS / margin / growth / "last quarter" → **numbers**: data → analysis → report
4. sentiment / tone / outlook → **sentiment**: data → classify → report
5. anything else → **text**: data → report

## Sample data shape (for the Data teammate)

`data/sample_amd.json` holds `company`, `revenue_quarterly` (fiscal labels like
`FY2025 Q2`, with `source`), `passages` (short filing sentences + `section` + `source`),
and a `quote`. Replace only the loading function in `data_agent` to use real SEC data;
nothing else changes.

## Status

Built in steps. Step 1 = skeleton (shared memory, base, llm+FakeLLM, sample data,
orchestrator, demo) with the 5 agents as placeholders. Later steps fill in the real
logic (numbers, report + citation check, sentiment, market + stored reuse) and tests.
