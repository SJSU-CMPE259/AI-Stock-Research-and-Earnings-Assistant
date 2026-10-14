# Nathan · NLP work log

## October 13, 2026 — Sprint 1 starter implemented with Codex

Branch: `feature/nathan-nlp-baseline`.

Completed implementation:

- `notebooks/p2_analysis_nlp.ipynb`: executed offline with spaCy's English model;
  shows sourced sentences, entity spans, evidence notes, annotation queue, and
  synthetic NER smoke evaluation.
- `ACSESS/finsight/backend/app/nlp/` and `agents/analysis.py`: reusable NLP pipeline,
  validated evidence contracts, stable IDs, aliases, safe annotation merging.
- `datasets/nlp/annotation_queue.csv`: three real sourced excerpts, zero human
  labels. Not a completed sentiment dataset.
- `docs/nlp_annotation_guide.md`: labeling conventions, reason IDs and review process.
- Root GitHub workflow for offline NLP regression tests.

Validation: notebook executed all six code cells; statistical-model CLI ran;
backend regression tests passed; lint passed for changed Python code. Repository-wide
lint still reports existing issues in unrelated SEC/UI/LLM files. The smoke example
containing the invented company "Acme Robotics" was missed by both NER variants,
demonstrating limited coverage rather than real-world filing accuracy.

Remaining human work: Nathan reviews the notebook, expands and labels the real
statement queue, and asks a teammate to review a subset. A7 still needs evaluation
on representative real annotated entity spans. Sentiment classification, financial
metrics, semantic citation checking and shared-memory integration are not finished.

Run instructions: `docs/nlp_baseline.md`. No measured sentiment accuracy or completed
human annotations are claimed.
