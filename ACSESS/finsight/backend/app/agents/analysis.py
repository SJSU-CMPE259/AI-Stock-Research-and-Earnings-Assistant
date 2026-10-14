"""Sprint 1 analysis: extract cited statements and entities from stored passages."""

import spacy
from spacy.language import Language

from backend.app.nlp.entities import extract_statements
from backend.app.nlp.schemas import AnalysisResult, EvidenceNote, Passage, stable_id


def analyze(passages: list[Passage], nlp: Language) -> AnalysisResult:
    """Return evidence notes for P3/P4; no sentiment or metrics are implied."""
    sources = {passage.passage_id: passage for passage in passages}
    statements = extract_statements(sources.values(), nlp)
    notes = tuple(
        EvidenceNote(
            note_id=stable_id("N", statement.statement_id),
            text=statement.text,
            source_ids=(statement.statement_id,),
        )
        for statement in statements
    )
    model = nlp.meta.get("name", "pipeline")
    version = nlp.meta.get("version", "unknown")
    return AnalysisResult(
        pipeline=f"spacy:{spacy.__version__}:{model}:{version}",
        sources=sources,
        statements=statements,
        notes=notes,
    )
