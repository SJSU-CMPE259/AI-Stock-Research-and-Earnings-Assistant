# Nathan's Sprint 1 NLP baseline

Start with `notebooks/p2_analysis_nlp.ipynb`. The notebook uses backend functions;
it does not duplicate the implementation or require running the API.

## Setup from the repository root

Use Python 3.11 or 3.12, not macOS's older system Python.

```bash
python3.12 -m venv .venv-nlp
source .venv-nlp/bin/activate
pip install -r ACSESS/finsight/requirements-nlp.txt
python -m spacy download en_core_web_sm
```

Select this environment as your notebook kernel. A project-local environment has
already been created during implementation. No SEC or LLM credentials are needed.
Setup downloads dependencies/model weights; subsequent notebook execution is offline.

## CLI

From `ACSESS/finsight`, using the same environment:

```bash
../../.venv-nlp/bin/python -m backend.scripts.analyze \
  --input backend/tests/fixtures/nlp/passages.json --output data/nlp
```

The CLI outputs `analysis.json` and a CSV queue. `--model rules` selects the explicit
rules-only alternative; `en_core_web_sm` remains the statistical default. Neither
path trains a model or generates sentiment ground truth.

## Modules and handoff

- `backend/app/nlp/schemas.py`: validated passage, statement, entity and note records.
- `backend/app/nlp/entities.py`: spaCy pipeline, alias normalization and sentence spans.
- `backend/app/agents/analysis.py`: `analyze(passages, nlp)` produces an `AnalysisResult`.
- `backend/app/nlp/annotations.py`: annotation queue export, merge and summary.
- `backend/app/nlp/evaluation.py`: exact entity span/label precision, recall and F1.

Pass stored, clean sections from P1 to `analyze`. P4 can persist the returned JSON;
P3 can resolve note -> statement -> passage to cite evidence. Shared memory and
orchestrator integration are not implemented yet. Numeric NER outputs such as
MONEY/PERCENT are text mentions, not authoritative XBRL financial metrics.

## Checks

From `ACSESS/finsight`:

```bash
../../.venv-nlp/bin/python -m pytest backend/tests/test_nlp.py -q
../../.venv-nlp/bin/ruff check backend/app/nlp backend/app/agents/analysis.py \
  backend/scripts/analyze.py backend/tests/test_nlp.py
```

Unit tests use spaCy's blank English pipeline plus company rules, so CI needs no
downloaded English model. The notebook additionally exercises the installed model.
The three synthetic NER examples demonstrate correctness/coverage limits only;
they are not measured accuracy on real filings. Annotate representative real
filing entity spans before reporting NER F1 to the class.

## Remaining work

A2: expand and manually label ~200 statements, review a subset and reserve an
evaluation split. A7: evaluate NER on real filing samples. Subsequent work:
FinBERT inference/fine-tuning/comparison, XBRL calculations, reason tagging,
contradiction checks, two-stage reporting and shared-memory integration.
