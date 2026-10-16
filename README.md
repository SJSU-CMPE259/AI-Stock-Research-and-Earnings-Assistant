# FinSight team project

The application is in `ACSESS/finsight/`; Anna's SEC exploration notebook is in
`notebooks/p1_edgar_xbrl.ipynb`.

Nathan's Sprint 1 NLP starter: [p2_analysis_nlp.ipynb](notebooks/p2_analysis_nlp.ipynb).
Setup and limitations: [NLP baseline](docs/nlp_baseline.md).
Human labeling instructions: [annotation guide](docs/nlp_annotation_guide.md).

## llm_pretrain
LLM Pretraining on Bridges-2 HPC


Structure
```
finsight/
├── README.md, pyproject.toml, .env.example, .gitignore
├── .github/workflows/ci.yml   # ruff + pytest on every PR
├── docs/                      # contracts.md, architecture, sprint notes
├── config/companies.yaml      # 10 tickers + CIKs
├── src
│   ├── memory/        (P4)    CAVM store on SQLite, schemas
│   ├── orchestrator/  (P4)    planner, step runner
│   ├── tools/         (P1)    edgar.py, xbrl.py, news.py
│   ├── rag/           (P1)    chunking, FAISS (Sprint 2)
│   ├── analysis/      (P2)    metrics, sentiment, NER
│   ├── report/        (P3)    writer, citation check
│   ├── api/           (P4)    FastAPI
│   └── eval/          (P5)    harness, baselines
├── ui/                (P5)
├── tests/                     # tests/fixtures/ = saved SEC JSON so CI never calls the SEC
├── notebooks/                 # scratch work
└── data/                      # gitignored: raw filings, caches
```

## Team and ownership

| Role | Owner | Folders |
|------|-------|---------|
| P1 Data | _name_ | `tools/`, `rag/` |
| P2 Analysis / NLP | _name_ | `analysis/` |
| P3 Report | _name_ | `report/` |
| P4 Agent core | _name_ | `memory/`, `orchestrator/`, `api/` |
| P5 UI, eval, deploy | _name_ | `ui/`, `eval/`, `.github/` |
