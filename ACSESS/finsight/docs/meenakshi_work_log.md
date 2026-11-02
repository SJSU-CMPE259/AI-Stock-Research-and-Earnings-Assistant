# FinSight — Meenakshi's work log & team handoff

Context for teammates: what I built, which branches it's on, how to run it, and the
new **Agent Demo** mode in the web app. Last updated 2026-10-15.

---

## TL;DR

- **Agent core** (multi-agent backend) → branch **`feature/agent-core`** → **PR #3**.
- **Beautiful UI rebuild** (in progress, paused) → branch **`feature/mistral-qa-ui`** (local, not pushed yet). Includes a **SEC parser fix** worth reviewing (see "For Anna").
- **Combined demo app** (portal + agents in one Streamlit app, with a new **Agent Demo** mode) → branch **`demo/portal-agents`** → pushed to team. **This is the one to run to see everything.**
- **LLM = Mistral-7B** self-hosted on **PSC Bridges-2 GPU** via Ollama, reached through `backend/app/llm.py` (`get_llm().complete`). Not a paid API.

---

## Repo quirk (read this first)

On my machine the git repo root is the **home directory**, and the project sits at
`ACSESS/finsight/`. The team repo has the same layout. Practical rule: **never
`git add -A`** — always stage specific files under `ACSESS/finsight/`. (If you cloned
the team repo normally, your root is the repo itself and this doesn't apply to you.)

---

## Branches & PRs

| Branch | What it is | Status |
|--------|-----------|--------|
| `feature/agent-core` | The `agent_core/` multi-agent core (self-contained, no existing files touched) | **PR #3 open** |
| `demo/portal-agents` | Portal **+** agent_core **+** the new **Agent Demo** mode in one app | Pushed to team (demo branch; not meant to merge to main as-is) |
| `feature/mistral-qa-ui` | The polished portal (hero, logos, Ask Mistral) **+** a merge of latest `main` **+** the SEC parser fix | Local only right now — the beautiful-UI rebuild is **paused** here |

PRs #1 (Nathan's NLP baseline) and #2 (edgar-xbrl) are already merged into `main`.
My agent-core work is **PR #3**. The parser fix lives on `feature/mistral-qa-ui`.

To get the combined demo:
```bash
git fetch
git checkout demo/portal-agents
cd ACSESS/finsight
.venv/bin/python -m streamlit run frontend/streamlit_app.py
# → http://localhost:8501
```

---

## The web app + the NEW "Agent Demo" mode

The Streamlit portal (`frontend/streamlit_app.py`) now has a **sidebar Mode switch**:

- **📄 Filing Q&A** — the original flow: pick company + filing type → *Fetch Latest
  Filing* (live SEC EDGAR) → ask a question → Mistral answers with `[S#]` citations.
- **🤖 Agent Demo** — runs the **multi-agent core** on bundled **AMD sample data**.
  You type a question (or pick a sample), toggle **Offline (FakeLLM) / real Mistral**,
  and watch the whole flow: **route → each agent (what it read / wrote / reused) →
  cited answer → shared-memory view**. Great for showing the professor the data flow.

> ⚠️ **Sample-data note:** the Agent Demo's AMD numbers are **made up** for
> development (revenue, passages, and the stock price `$168.42` are all fake — every
> answer ends with "(sample data)"). `market_agent` is wired to read that sample now
> and be swapped for a **live price API later**.

---

## What the agent core is (`agent_core/`)

Agents never call each other — they read/write **named items in one shared memory**
(the CAVM idea). An orchestrator routes each question with **simple Python rules**
(Mistral-7B is weak at tool-calling), runs the agents in order, and returns a trace.

- `memory.py` — `SharedMemory` (put/get/has/list/describe/save) + `Conversation` (last 3 Q/A pairs).
- `base.py` — `Agent` base class.
- `llm.py` — `MistralLLM` (wraps `get_llm()`, with a system→user fallback) + `FakeLLM` (offline).
- `orchestrator.py` — routing (refuse / real_time / numbers / sentiment / text), stored-data reuse, 60s quote expiry, trace.
- `agents/<name>/agent.py` — data, analysis (QoQ/YoY in Python), classify (Mistral one-word sentiment), report (Mistral answer + citation/number check + template fallback), market (real_time quote).
- `data/sample_amd.json` — the sample shape. **Data teammate:** replace only
  `data_agent.load_company_data()` with real SEC tools; nothing else changes.
- `run_demo.py` — 5-question CLI demo: `python -m agent_core.run_demo [--fake]` (run from `ACSESS/finsight/`).
- `tests/` — 11 offline tests (FakeLLM, no network): `pytest agent_core/tests/ -q`.

---

## LLM setup — Mistral on PSC (how to bring it up)

Mistral runs on a PSC Bridges-2 **V100** GPU via **Ollama**, tunneled to `localhost:8000`.
`.env` uses provider `remote`, `REMOTE_LLM_URL=http://localhost:8000/v1`, `LLM_MODEL=mistral`.

Bring it up when you need real answers (the app also runs fully **offline** without it):
```bash
# 1) On PSC: get a GPU node
ssh <you>@bridges2.psc.edu
salloc -p GPU-small -N 1 --gres=gpu:1 -t 04:00:00      # note the node, e.g. v001
# 2) On that node (everything in /tmp to dodge the home-dir quota):
cd /tmp
curl -L https://github.com/ollama/ollama/releases/download/v0.40.3/ollama-linux-amd64.tar.zst -o ollama.tar.zst
zstd -d ollama.tar.zst -o ollama.tar && mkdir -p ollama-bin && tar -xf ollama.tar -C ollama-bin
export OLLAMA_MODELS=/tmp/ollama OLLAMA_HOST=0.0.0.0:8000 LD_LIBRARY_PATH=/tmp/ollama-bin/lib/ollama
/tmp/ollama-bin/bin/ollama serve
# 3) Second shell inside the job (NOT a plain `ssh v001` — /tmp is per-job):
srun --jobid=<JOBID> --pty bash
OLLAMA_HOST=0.0.0.0:8000 /tmp/ollama-bin/bin/ollama pull mistral
# 4) On your Mac: open the tunnel (leave it running)
ssh -L 8000:<NODE>:8000 <you>@bridges2.psc.edu
# 5) Check:
curl http://localhost:8000/v1/models      # should list "mistral"
```
Note: Bridges-2 GPU nodes are V100 (compute 7.0). Modern vLLM/torch wheels fail on
them ("no kernel image"); Ollama's bundled `cuda_v12` build works. The home dir has a
tight quota — keep big files in `/tmp` (per-job) or `/ocean`.

---

## For Anna — SEC parser fix (`backend/app/tools/sections.py`)

The parser took the **first** regex match for each Item, which is the **table-of-contents
line**, so only tiny ~40-char stubs were returned and Item 1A was effectively missing
(Mistral kept saying risk factors "weren't in the excerpts"). Fix: scan **every**
occurrence and keep the one with the most body text (≥ `MIN_BODY_CHARS=2000`),
skipping the ToC stub. Added `backend/tests/test_sections_toc.py`.

Verified on NVDA's latest 10-K — Item 1A now extracts **114,916 chars** (was ~40):

| Section | Chars |
|---------|------:|
| Item 1 | 48,241 |
| Item 1A | 114,916 |
| Item 7 | 34,154 |
| Item 7A | 4,253 |

On `feature/mistral-qa-ui` (commit `152c3fc`). Please review.

---

## Files I worked on

- **`agent_core/`** — entire new folder (multi-agent core + demo + tests + sample data).
- **`frontend/streamlit_app.py`** — hero banner, company logos, Ask Mistral panel (portal); the **Agent Demo** mode (on `demo/portal-agents`).
- **`backend/app/tools/sections.py`** — parser ToC fix (for Anna) + **`backend/tests/test_sections_toc.py`**.
- **`.env` / `.env.mistral-example` / `test_mistral.py`** — Mistral/PSC config + connectivity test.

I did **not** change anyone else's files (Nathan's `backend/app/nlp/*`, notebooks, docs
are untouched). `backend/app/llm.py` and `settings.py` are unchanged.

---

## What's next / paused

- **UI rebuild is paused** on `feature/mistral-qa-ui` (Steps: split UI into modules,
  offline-proof status bar + Demo mode, grounded answers with citation chips, pipeline
  stepper + Altair charts). Resume anytime.
- **Live price API** for `market_agent` (replace the sample `$168.42`) — future.
- **Eval harness** (Jev vs. Mistral setups) — planned, not started.
