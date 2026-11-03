# finsight.rag: retrieval over filing text (vector DB)

Filing paragraphs live in a **Qdrant** vector database. Each paragraph has two vectors: an embedding for meaning and BM25 weights for keywords. Search runs in one of three modes, which are also the three arms of the RQ1 ablation: `sparse`, `dense` or `hybrid` (the default).

The SQLite datastore (`data/finsight.db`) is unchanged and still holds filings, XBRL numbers and fiscal labels. The RAG code only *reads* it, to get the list of filings.

## What goes where

| Question needs… | Comes from | RAG? |
|---|---|---|
| A reported number (revenue, EPS, cash) | `datastore.api.get_metric` (XBRL, SQLite) | No |
| Price now / recent news | `tools.prices`, `tools.news` | No |
| What a filing *says* (risks, MD&A, legal, controls, business) | `rag.retrieve` → vector DB | **Yes** |
| What an earnings 8-K says (press release) | `rag.retrieve(..., forms=("8-K",))` | **Yes** |

## Install

Add to `requirements.txt`:

```
qdrant-client>=1.12
fastembed>=0.4          # or fastembed-gpu on a CUDA machine
# openai>=1.0           # only if you use FINSIGHT_EMBED_MODEL=openai:...
```

`fastembed` runs the embedding model with ONNX on CPU, with no PyTorch needed. The default model, `BAAI/bge-small-en-v1.5` (384-dim), downloads once, about 70 MB.

## Where the vector DB runs

| Setting | Use it for |
|---|---|
| *(nothing set)* | Embedded Qdrant in `data/qdrant/`: notebooks and scripts. **Only one process can open it at a time.** |
| `FINSIGHT_QDRANT_URL=http://localhost:6333` | Qdrant server: the Streamlit app, several processes at once, and scaling past 10 companies |
| `FINSIGHT_QDRANT_URL` + `FINSIGHT_QDRANT_API_KEY` | Qdrant Cloud (free tier available) |

Run a local server with Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/data/qdrant_server:/qdrant/storage qdrant/qdrant
```

The code is the same in every case. Switching is a matter of setting the URL and running `--rebuild` once.

Other settings:
- `FINSIGHT_EMBED_MODEL`: any fastembed text model, for example `BAAI/bge-base-en-v1.5`. Use `hash` for CI and tests: no download, but no real meaning either.
- `FINSIGHT_QDRANT_COLLECTION`: defaults to `finsight_passages`.

## Speed: making embedding faster

Embedding is the slow step: every paragraph (~300 tokens) goes through the model once. Already built in:

- Paragraphs from several filings are embedded together, 2,048 at a time (`FINSIGHT_EMBED_CHUNK`). Big batches are much faster than one 10-Q at a time.
- On a CPU, big batches are split across all cores (`FINSIGHT_EMBED_PARALLEL=0`, the default; `off` turns it off).
- Uploads to Qdrant run in the background while the next batch is embedded.
- Progress prints after each batch, with paragraphs per second.

| Option | How | When |
|---|---|---|
| **Index once, share it** | Build on one machine into Qdrant Cloud; teammates only query | Always: only one person waits |
| **GPU** | `pip install fastembed-gpu` (instead of `fastembed`), `FINSIGHT_EMBED_CUDA=1` | On the PSC GPU machine |
| **Skip sections** | `FINSIGHT_RAG_SKIP_TOPICS=exhibits,summary`, then `--force` | Item 15 is large. For some companies (e.g. NVIDIA, Oracle) it holds the financial-statement notes, so you lose those |
| **OpenAI embeddings** | `pip install openai`, `OPENAI_API_KEY=...`, `FINSIGHT_EMBED_MODEL=openai:text-embedding-3-small`, then `--rebuild` | Slow laptop and no GPU. ~$0.02 per 1M tokens; every question then calls OpenAI too |

Only new filings are embedded on later runs. Avoid `--force` and `--rebuild` unless you changed the splitter, the model or the skip list.

## Build the index

```bash
python scripts/build_db.py                            # existing: SEC -> SQLite + downloaded filings
python scripts/build_rag_index.py --press-releases    # fetch 8-K Exhibit 99.1, then embed + index
python scripts/build_rag_index.py --ask "How do export limits affect Nvidia?" --ticker NVDA --mode dense
python scripts/build_rag_index.py --rebuild           # after changing the embedding model
```

Re-running only indexes new filings. A filing is re-embedded when `SPLITTER_VERSION` changes or when you pass `--force`.

## Use it from an agent

```python
from finsight.rag import retrieve, to_cavm

r = retrieve("What new risks did Apple flag?", "AAPL")                  # hybrid by default
r = retrieve("...", "AAPL", topics=["risk_factors"], search_mode="dense")  # router-picked topic
evidence = to_cavm(r["evidence"])     # [{"id": "S1", "text", "citation", ...}] -> shared memory
```

`retrieve()` is unchanged from step 1:

1. Picks the section(s) the question needs.
2. Returns them whole if they fit `budget_tokens` (`mode="section"`).
3. Otherwise searches inside them with `search_mode` and keeps the best paragraphs (`mode="search"`).

## Retrieval quality settings

| Setting | What it does |
|---|---|
| `search_mode="rerank"` / `--mode rerank` | Hybrid top 30, re-scored by a cross-encoder (`rerank.py`). Default reranker `Xenova/ms-marco-MiniLM-L-6-v2` (~80 MB); try `FINSIGHT_RERANK_MODEL=BAAI/bge-reranker-base` |
| `FINSIGHT_SEARCH_MODE=rerank` | Make rerank the default for `retrieve()` once the eval shows it helps |
| `FINSIGHT_QUERY_PREFIX` | Instruction added to questions for bge models (on by default; `""` turns it off) |

Always on:
- **No one-line fragments.** Paragraphs under 30 words are merged: a risk factor's title joins its explanation, and a 10-K's risk summary bullets become one passage (splitter v3).
- **No duplicates.** Search results drop repeated paragraphs, which 10-Qs copy from earlier filings, and keep the newest copy.
- **BM25 ignores filler words** found in nearly every paragraph ("company", "business", "may"…).

## RQ1 ablation

```python
from finsight.rag.eval import evaluate
rows, summary = evaluate(cases, k=5)     # hit@5, recall@5, MRR for sparse / dense / hybrid / rerank
```

Section 7 of `notebooks/rag_walkthrough.ipynb` runs it on 10 generic questions per company. It also helps you turn regex gold into hand-checked `gold_ids`.

## Scale

At 384 dimensions, a vector is about 1.5 KB.

| Companies | Paragraphs (≈5.6K per company, 3 years) | Vectors | Where |
|---|---|---|---|
| 10 | ~56K | ~90 MB | Embedded is fine |
| 100 | ~560K | ~0.9 GB | Qdrant server |
| 500 (S&P 500) | ~2.8M | ~4.3 GB (~1.1 GB with scalar quantization) | Qdrant server or Cloud |

Filters on ticker, form, topic and accession keep each search small. In server mode those fields are indexed automatically.

## Moving from the SQLite version

The step-1 tables in `finsight.db` are no longer used. Drop them once, or leave them in place; they're harmless:

```sql
DROP TABLE IF EXISTS passages_fts; DROP TABLE IF EXISTS passages;
DROP TABLE IF EXISTS filing_sections; DROP TABLE IF EXISTS rag_index_log;
```
