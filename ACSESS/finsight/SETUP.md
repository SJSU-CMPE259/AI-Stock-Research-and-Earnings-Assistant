# FinSight: Setup & Getting Started

Welcome to **FinSight** — a financial research assistant that answers questions about US public company SEC filings with **cited sources**.

## 📋 Prerequisites

- **Python 3.11+**
- **Git**
- **An API key** (optional for development):
  - Anthropic (for `LLM_PROVIDER=anthropic`)
  - OpenAI (for `LLM_PROVIDER=openai`)
  - Or use vLLM on Bridges-2 PSC (remote LLM)

## 🚀 Quick Start (5 minutes)

### 1. Clone the repo

```bash
git clone https://github.com/SJSU-CMPE259/AI-Stock-Research-and-Earnings-Assistant.git
cd AI-Stock-Research-and-Earnings-Assistant/ACSESS/finsight
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies (minimal)

```bash
pip install -r requirements-core.txt
```

This installs only the essentials:
- `requests` — SEC API calls
- `beautifulsoup4` — HTML parsing
- `pydantic`, `pydantic-settings` — Configuration
- `python-dotenv` — Environment variables

### 4. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
SEC_USER_AGENT=Your Full Name your.email@sjsu.edu
LLM_PROVIDER=remote
REMOTE_LLM_URL=http://localhost:8000/v1
```

**If using Anthropic or OpenAI instead:**

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-5
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Launch the portal

```bash
streamlit run frontend/streamlit_app.py
```

Open: **http://localhost:8501**

---

## 📊 What You Can Do Now

✅ **Fetch SEC Filings**
- Select company (NVDA, AAPL, etc. — more coming)
- Choose filing type (10-K, 10-Q)
- Automatically downloads from SEC EDGAR

✅ **View Parsed Sections**
- Risk Factors, MD&A, Financial tables
- Extracted intelligently from HTML
- Fully searchable and copyable

✅ **Offline Development**
- All data cached locally (`data/raw/`)
- No external API calls (except SEC)
- Works without LLM connection

---

## 🔧 Development Setup (Full)

For contributing to the project:

```bash
# Install all dependencies
pip install -r requirements.txt

# This includes:
# - FastAPI + uvicorn (backend)
# - sentence-transformers + FAISS (vector search)
# - pytest (testing)
# - ruff (linting)
# - Streamlit (frontend)
```

### Run the backend API

```bash
cd backend
uvicorn app.main:app --reload
```

Open: **http://localhost:8000/docs** (Swagger UI)

### Run tests

```bash
pytest backend/tests/ -v
```

### Lint code

```bash
ruff check backend/ frontend/
```

---

## 🔄 Project Structure

```
finsight/
├── CLAUDE.md              # Architecture & conventions
├── README.md              # Quick overview
├── .env.example           # Template for env vars
├── requirements.txt       # All dependencies
├── requirements-core.txt  # Minimal (SEC tools only)
│
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI routes (coming)
│   │   ├── settings.py    # Pydantic configuration
│   │   ├── llm.py         # LLM wrapper (Anthropic/OpenAI/remote)
│   │   ├── tools/
│   │   │   ├── sec_client.py  # Throttled HTTP client
│   │   │   ├── edgar.py       # SEC EDGAR API
│   │   │   ├── sections.py    # HTML parsing & extraction
│   │   │   └── xbrl.py        # Quantitative data (coming)
│   │   └── rag/           # Retrieval (coming in Milestone 2)
│   ├── scripts/
│   │   └── fetch.py       # CLI demo
│   └── tests/             # Unit tests
│
├── frontend/
│   └── streamlit_app.py   # Web portal
│
├── data/
│   ├── raw/               # Cached SEC filings
│   └── finsight.db        # SQLite (coming)
│
└── eval/
    └── retrieval_eval.py  # Evaluation harness (coming)
```

---

## 📝 Common Tasks

### Fetch NVDA's latest 10-K (CLI)

```bash
python -m backend.scripts.fetch
```

Output: Filing metadata + extracted sections

### Test SEC tools

```bash
pytest backend/tests/test_sec_tools.py -v
```

### Check code quality

```bash
ruff check backend/
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Example |
|----------|----------|---------|---------|
| `SEC_USER_AGENT` | ✅ Yes | — | `John Doe john@sjsu.edu` |
| `LLM_PROVIDER` | Yes | `remote` | `anthropic`, `openai`, `remote` |
| `LLM_MODEL` | Yes | `mistralai/Mistral-7B-Instruct-v0.1` | `claude-opus-5` |
| `ANTHROPIC_API_KEY` | If `anthropic` | — | `sk-ant-...` |
| `OPENAI_API_KEY` | If `openai` | — | `sk-...` |
| `REMOTE_LLM_URL` | If `remote` | `http://localhost:8000/v1` | vLLM URL |
| `DATA_DIR` | No | `./data` | Path for caching |
| `DEBUG` | No | `false` | `true` or `false` |

---

## 🚨 Troubleshooting

**"ModuleNotFoundError: No module named 'streamlit'"**
```bash
pip install streamlit
```

**"SEC_USER_AGENT not set"**
→ Edit `.env` with your name and email (SEC requirement)

**"Connection refused at localhost:8501"**
→ Streamlit is not running. Run: `streamlit run frontend/streamlit_app.py`

**"Couldn't find a tree builder with the features you requested: lxml"**
```bash
pip install lxml
```

---

## 📚 Next Steps

**Milestone 1 (Current):** ✅ SEC data tools + Streamlit portal
**Milestone 2 (Next):** Ingestion & retrieval (FAISS + BM25)
**Milestone 3:** LLM Q&A with citations
**Milestone 4:** Full dashboard + team features

---

## 🤝 Team Workflow

1. **Clone** the repo and create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Work** on your agent/module (Data, Analysis, Classify, Report)

3. **Test** locally:
   ```bash
   pytest backend/tests/ -v
   ruff check .
   ```

4. **Push** and create a pull request to `main`

---

## 🔗 Resources

- **CLAUDE.md** — Full architecture & conventions
- **README.md** — Project overview
- **SEC EDGAR** — https://www.sec.gov/edgar
- **GitHub** — https://github.com/SJSU-CMPE259/AI-Stock-Research-and-Earnings-Assistant

---

## 📧 Questions?

Ask the team! FinSight is collaborative. Each agent (Data, Analysis, Classify, Report) can be built independently and plugged in without breaking the skeleton.

**Happy researching! 🚀**
