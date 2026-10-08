# FinSight: Financial Q&A Research Assistant

Ask questions about US public companies and get cited answers from SEC filings.

**Status:** Walking skeleton (Milestones 0–4)  
**Supported companies:** NVDA (+ 9 more coming)

## Quick Start

### 1. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your SEC_USER_AGENT and LLM API key
```

### 2. Run

```bash
# Terminal 1: Backend API
cd backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### 3. Ingest data (one-time)

Before asking questions, ingest NVDA's filings:

```bash
python -m backend.scripts.ingest --ticker NVDA
```

## Example

**Question:** "What major risks does NVIDIA identify in its latest annual filing?"

**Answer:**
> NVIDIA faces significant export control risks, particularly restrictions on advanced AI chips to certain countries [S1]. The company is heavily dependent on TSMC for semiconductor manufacturing, creating supply chain concentration risk [S2]. Competition in the data center market remains intense, with established competitors and new entrants continuously improving products [S3].

**Sources:**
- [S1] NVIDIA 10-K (2026-01-25), Item 1A Risk Factors
- [S2] NVIDIA 10-K (2026-01-25), Item 1 Business
- [S3] NVIDIA 10-K (2026-01-25), Item 1A Risk Factors

## Features

- ✅ SEC filing retrieval (10-K, 10-Q)
- ✅ Hybrid search (embeddings + BM25)
- ✅ Cited answers (every claim has [S#])
- ✅ Source panel with SEC links
- ✅ Guardrail against buy/sell advice
- 🚀 (Coming) Quantitative analysis, sentiment, deep research

## Development

See [CLAUDE.md](./CLAUDE.md) for:
- Architecture & data flow
- Tech stack decisions
- Module breakdown
- Testing and evaluation
- Git workflow

## Testing

```bash
# Unit tests
pytest backend/tests/ -v

# Retrieval evaluation
python eval/run_retrieval_eval.py

# Live integration test (requires API keys)
pytest -m network
```

## Docker

```bash
docker compose up
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

## API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What risks does NVDA face?","ticker":"NVDA"}'
```

Response:
```json
{
  "answer": "...",
  "citations": [
    {
      "id": "S1",
      "chunk_id": "NVDA_10-K_2026-01-25_ITEM1A_001",
      "ticker": "NVDA",
      "form": "10-K",
      "report_date": "2026-01-25",
      "section": "Item 1A",
      "snippet": "...",
      "url": "https://www.sec.gov/..."
    }
  ],
  "run_id": "...",
  "latency_ms": 234
}
```

## Project

**Team:** SJSU CMPE 259 (NLP)  
**Repo:** https://github.com/SJSU-CMPE259/AI-Stock-Research-and-Earnings-Assistant

---

*For research and educational purposes only. Not financial advice.*
