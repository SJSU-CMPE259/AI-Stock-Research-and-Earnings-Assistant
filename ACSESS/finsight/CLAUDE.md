# FinSight: Financial Q&A Research Assistant

**Team:** SJSU CMPE 259 (NLP) — AI Stock Research and Earnings Assistant  
**Status:** Walking skeleton (Milestone 0–4)  
**Last updated:** 2026-10-08

## Overview

FinSight is a multi-agent financial research assistant that answers questions about US public companies using SEC filings (10-K, 10-Q). Every factual claim in an answer carries a citation `[S1]` linked to a specific section, filing date, and SEC URL. The system never invents numbers or financial advice.

**Current scope (this session):**
- End-to-end Q&A pipeline: company selection → question → cited answer
- NVDA only (team will add 9 more companies in future PRs)
- Retrieval-based answers from SEC filings (no quantitative analysis yet)
- Streamlit UI + FastAPI backend + SQLite storage

## Architecture

### Data flow

```
User Question
    ↓
[Orchestrator] (resolve company, guardrail check)
    ↓
[Retriever] (FAISS + BM25 hybrid search over filing sections)
    ↓
[Memory] (store sources as S1, S2, ... in session)
    ↓
[Report Agent] (LLM writes answer with citations)
    ↓
[Citation Check] (validate [S#] references, retry once)
    ↓
[Frontend] (display answer + sources panel)
```

### Key modules

**Backend (`backend/app/`)**

- `settings.py` — Pydantic config from .env; fails loudly if SEC_USER_AGENT or API key missing
- `main.py` — FastAPI routes: `/health`, `/companies`, `/ask`, `/feedback`, `/runs/{run_id}`
- `orchestrator.py` — QA pipeline: guard rails → retrieve → store in memory → generate answer
- `llm.py` — Thin wrapper over Anthropic or OpenAI SDK; logs calls, supports caching, retries
- `memory.py` — Named-variable store per session; sessions persist ticker context
- `checks.py` — Citation validator and financial-advice guardrail
- `db.py` — SQLAlchemy models and SQLite initialization
- `agents/base.py`, `agents/report.py` — Agent ABC and answer-writing agent
- `agents/{data,analysis,classify}.py` — Stub agents for future phases

**SEC Data Tools (`backend/app/tools/`)**

- `sec_client.py` — Throttled requests.Session (max 8 req/s), disk cache, exponential backoff
- `edgar.py` — Ticker ↔ CIK, filing lists, filing downloads
- `sections.py` — HTML → text, extract 10-K/10-Q Items by name
- `xbrl.py` — Quantitative facts (revenues, ratios) from company facts API
- `news.py` — Stub for Finnhub integration

**RAG (`backend/app/rag/`)**

- `chunk.py` — Paragraph-based chunking with configurable word targets and overlap
- `ingest.py` — Idempotent ingestion of 10-K/10-Q sections into SQLite
- `index.py` — Build and persist FAISS and BM25 indices per ticker
- `retrieve.py` — Hybrid retrieval (FAISS + BM25) with reciprocal rank fusion

**Frontend (`frontend/streamlit_app.py`)**

- Chat UI with company selector
- Question input
- Cited answer display
- Sources panel with filing metadata and SEC links
- Thumbs up/down feedback
- Always-visible research disclaimer

**Scripts (`backend/scripts/`)**

- `fetch.py` — CLI to download and parse NVDA filings
- `ingest.py` — CLI to index all companies into SQLite + FAISS/BM25
- `search.py` — CLI to test retrieval

**Evaluation (`eval/`)**

- `retrieval_questions.yaml` — ~10 NVDA benchmarks with expected sections
- `run_retrieval_eval.py` — Computes Recall@k and MRR for vector vs. hybrid retrieval

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Backend | FastAPI + uvicorn | Async, ASGI, modern Python |
| Database | SQLite + SQLAlchemy | Simple, file-based, good for prototypes |
| SEC data | requests + BeautifulSoup | Direct SEC API; no external service lock-in |
| Embeddings | sentence-transformers | "all-MiniLM-L6-v2" by default; configurable |
| Vector search | FAISS (CPU) | IndexFlatIP on L2-normalized vectors (cosine) |
| Keyword search | rank_bm25 | BM25Okapi for lexical matching |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) | Combines FAISS + BM25 scores |
| LLM | Anthropic or OpenAI | Provider agnostic; env-driven selection |
| Frontend | Streamlit | Rapid iteration, chat UI, no React needed yet |
| Testing | pytest | Unit tests only; no network calls in CI |
| Linting | ruff | Fast, Python-native |

## Environment Setup

### 1. Clone and install

```bash
cd finsight
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure .env

```bash
cp .env.example .env
# Edit .env:
# - SEC_USER_AGENT="Your Name your.email@sjsu.edu"
# - LLM_PROVIDER and LLM_MODEL
# - ANTHROPIC_API_KEY or OPENAI_API_KEY
```

Validation happens at startup:
```python
from backend.app.settings import get_settings
settings = get_settings()  # Fails if SEC_USER_AGENT or API key missing
```

### 3. Run tests

```bash
pytest backend/tests/ -v
```

Unit tests never touch the network. Live tests marked `@pytest.mark.network` (skipped by default):
```bash
pytest -m network  # Requires .env with valid API keys
```

## Running the Stack

### Backend only (API)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs (Swagger UI)
```

### Frontend + Backend (full stack)

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
streamlit run streamlit_app.py
# → http://localhost:8501
```

### With Docker

```bash
docker compose up
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

## Development Workflow

### Before writing code

1. Check the brief for the milestone target and deliverables
2. Inspect real API responses (fixtures in `backend/tests/fixtures/`)
3. Write tests before or alongside code (test-driven development)

### Code standards

- **Python 3.11+** with type hints (`str | None`, `list[T]`, etc.)
- **Short docstrings**: one-liner describing what/why, not obvious WHAT
- **Logging, not print**: use `logging` module in library code
- **Small functions**: ~10 lines, one responsibility each
- **Pydantic models** for all API schemas and data classes
- **No comments** unless the WHY is non-obvious (workaround, subtle invariant)
- **DRY only if 3+ similar lines**: premature abstraction is worse than duplication
- **No error handling for impossible scenarios**: trust frameworks and APIs

### Secrets and data

- `.env` only; never commit
- Data files (`data/raw/`, `data/index/`, `finsight.db`) in `.gitignore`
- Fixtures in `backend/tests/fixtures/` as small, trimmed JSON/HTML snippets

### Git workflow

After each milestone, run tests and commit:

```bash
# Run all checks
pytest backend/tests/ -v
ruff check backend/ frontend/

# Commit
git add -A
git commit -m "feat: [Milestone N] brief summary

- Bullet 1
- Bullet 2
- Bullet 3

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

## Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| 0 | Repo scaffold, .env, CLAUDE.md, CI | In progress |
| 1 | SEC tools (edgar.py, sections.py, xbrl.py) | —— |
| 2 | Ingestion & retrieval (FAISS, BM25, RRF) | —— |
| 3 | Orchestrator, LLM wrapper, API, memory, citation checks | —— |
| 4 | Streamlit UI + Docker Compose | —— |

**Not in this session (stubs only):**
- Quantitative analysis (xbrl.py results, charts)
- Deep research reports, sentiment classification
- Scheduler, alerts, MCP server
- Jev, Deep Search, translation, voice
- User accounts, React frontend, cloud deployment

## API Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET | `/health` | `{status: "ok"}` |
| GET | `/companies` | `[{ticker, name, enabled}]` |
| POST | `/ask` | `{answer, citations, route, run_id, latency_ms, warnings}` |
| POST | `/feedback` | `{id}` (stores rating + comment) |
| GET | `/runs/{run_id}` | Memory entries and metadata (debug) |

See `backend/app/main.py` for request/response schemas.

## Testing

### Unit tests

```bash
pytest backend/tests/ -v --tb=short
```

Located in `backend/tests/`:
- `test_sec_tools.py` — edgar, sections, xbrl parsers with fixtures
- `test_rag.py` — chunking, embedding, retrieval logic
- `test_orchestrator.py` — Q&A pipeline with fake LLM/retriever
- `test_api.py` — FastAPI routes with TestClient
- `conftest.py` — Shared fixtures (mock SEC responses, in-memory DB)

### Retrieval evaluation

```bash
python eval/run_retrieval_eval.py
# Output: Recall@3, Recall@6, MRR for vector and hybrid retrieval
```

Evaluation questions in `eval/retrieval_questions.yaml`.

### Live integration test (requires .env with real API keys)

```bash
pytest -m network
```

## Debugging

### Backend logs

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

LLM calls are logged to `llm_calls` SQLite table (model, latency, tokens).

### SQLite queries

```bash
sqlite3 data/finsight.db
sqlite> SELECT * FROM llm_calls;
sqlite> SELECT * FROM memory WHERE run_id = '...';
```

### API responses

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What risks does NVDA face?","ticker":"NVDA"}'
```

## Future Phases (Team Branches)

Each team will open a separate branch for their agent phase:

1. **Data Agent** (filings, news, earnings calls) → `feature/data-agent`
2. **Analyst Agent** (comparative analysis, trends) → `feature/analysis-agent`
3. **Classify Agent** (sentiment, risk scoring) → `feature/classify-agent`
4. **Quantitative Pipeline** (xbrl, financial ratios, charts) → `feature/quant-pipeline`
5. **Deep Research** (multi-step investigation, synthesis) → `feature/deep-research`
6. **MCP Server** (Claude integration) → `feature/mcp-server`

Each branch must:
- Not break the Q&A skeleton
- Add tests for new functionality
- Update CLAUDE.md with new modules
- Merge to `main` only after review

## Repo Structure

```
finsight/
├── CLAUDE.md (this file)
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── config/
│   └── companies.yaml
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── settings.py
│   │   ├── db.py
│   │   ├── orchestrator.py
│   │   ├── memory.py
│   │   ├── llm.py
│   │   ├── checks.py
│   │   ├── prompts/
│   │   │   └── qa.md
│   │   ├── agents/
│   │   │   ├── base.py
│   │   │   ├── report.py
│   │   │   ├── data.py, analysis.py, classify.py (stubs)
│   │   ├── tools/
│   │   │   ├── sec_client.py
│   │   │   ├── edgar.py
│   │   │   ├── sections.py
│   │   │   ├── xbrl.py
│   │   │   └── news.py (stub)
│   │   └── rag/
│   │       ├── chunk.py
│   │       ├── ingest.py
│   │       ├── index.py
│   │       └── retrieve.py
│   ├── scripts/
│   │   ├── fetch.py
│   │   ├── ingest.py
│   │   └── search.py
│   └── tests/
│       ├── conftest.py
│       ├── test_*.py
│       └── fixtures/
├── frontend/
│   └── streamlit_app.py
├── eval/
│   ├── retrieval_questions.yaml
│   └── run_retrieval_eval.py
├── data/ (gitignored)
│   ├── raw/
│   ├── index/
│   └── finsight.db
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Contact

**Project lead:** SJSU CMPE 259 instructors  
**Repo:** https://github.com/SJSU-CMPE259/AI-Stock-Research-and-Earnings-Assistant

---

*Last updated: 2026-10-08*
