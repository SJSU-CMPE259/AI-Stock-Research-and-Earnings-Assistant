"""Provenance, normalization, and safe annotation workflow regression tests."""

import csv

import pytest
from pydantic import ValidationError

from backend.app.agents.analysis import analyze
from backend.app.nlp.annotations import annotation_summary, export_annotations
from backend.app.nlp.entities import build_pipeline
from backend.app.nlp.evaluation import evaluate_entities
from backend.app.nlp.schemas import AnalysisResult, EvidenceNote, Passage


@pytest.fixture(scope="module")
def nlp():
    return build_pipeline(None)


def passage(text="  NVIDIA discussed NVDA.\n\nApple mentioned AMD.  ", synthetic=False):
    return Passage(
        ticker="NVDA",
        source_type="synthetic" if synthetic else "filing",
        source_url="https://www.sec.gov/Archives/example.htm",
        accession="0001045810-25-000023",
        form="10-K",
        period_end="2025-01-26",
        section="Item 7",
        text=text,
    )


def test_offsets_aliases_and_note_links(nlp):
    source = passage()
    result = analyze([source], nlp)
    assert len(result.statements) == 2
    entities = [ent for statement in result.statements for ent in statement.entities]
    assert [(e.text, e.label, e.canonical_ticker) for e in entities] == [
        ("NVIDIA", "ORG", "NVDA"),
        ("NVDA", "TICKER", "NVDA"),
        ("Apple", "ORG", "AAPL"),
        ("AMD", "TICKER", "AMD"),
    ]
    for statement, note in zip(result.statements, result.notes):
        assert source.text[statement.start : statement.end] == statement.text
        assert note.source_ids == (statement.statement_id,)
        assert note.text == statement.text


def test_stable_ids_and_duplicate_passages(nlp):
    p = passage()
    first = analyze([p], nlp)
    assert first.model_dump() == analyze([p, p], nlp).model_dump()
    changed = p.model_copy(update={"text": p.text + " New sentence."})
    assert p.passage_id != changed.passage_id


def test_longest_company_alias_and_case(nlp):
    result = analyze(
        [passage("nvidia corporation mentioned Microsoft Corporation.")], nlp
    )
    entities = result.statements[0].entities
    assert [(e.text, e.canonical_ticker) for e in entities] == [
        ("nvidia corporation", "NVDA"),
        ("Microsoft Corporation", "MSFT"),
    ]


def test_real_sources_and_whitespace_are_validated():
    with pytest.raises(ValidationError):
        Passage(ticker="NVDA", source_type="filing", section="Item 7", text="Test.")
    with pytest.raises(ValidationError):
        passage(" \n ")


def test_annotations_preserve_human_labels_and_old_rows(nlp, tmp_path):
    target = tmp_path / "annotations.csv"
    first = analyze([passage("NVIDIA discussed revenue.")], nlp)
    export_annotations(first, target)
    with target.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0].update(
        gold_sentiment="neutral",
        gold_reasons="revenue",
        annotator="Nathan",
        review_status="reviewed",
    )
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    export_annotations(first, target)
    second = analyze([passage("Apple discussed competition.")], nlp)
    assert export_annotations(second, target) == 2
    assert annotation_summary(target) == {
        "total": 2,
        "labeled": 1,
        "reviewed": 1,
        "usable_reviewed": 1,
    }
    assert "Nathan" in target.read_text()


def test_annotation_conflicting_evidence_rejected(nlp, tmp_path):
    target = tmp_path / "annotations.csv"
    result = analyze([passage()], nlp)
    export_annotations(result, target)
    contents = target.read_text().replace("NVIDIA discussed NVDA.", "Altered evidence.")
    target.write_text(contents)
    with pytest.raises(ValueError, match="evidence changed"):
        export_annotations(result, target)
    assert target.read_text() == contents


def test_synthetic_examples_excluded_from_sentiment_queue(nlp, tmp_path):
    result = analyze([passage(synthetic=True)], nlp)
    target = tmp_path / "annotations.csv"
    assert export_annotations(result, target) == 0
    assert annotation_summary(target)["total"] == 0


def test_unknown_note_and_corrupt_spans_rejected(nlp):
    result = analyze([passage()], nlp)
    data = result.model_dump()
    data["notes"] = [
        EvidenceNote(
            note_id="Nbad", text="Unsupported", source_ids=("Sbad",)
        ).model_dump()
    ]
    with pytest.raises(ValidationError, match="existing statement IDs"):
        AnalysisResult.model_validate(data)
    data = result.model_dump()
    data["statements"][0]["entities"][0]["end"] = 999
    with pytest.raises(ValidationError, match="outside the statement"):
        AnalysisResult.model_validate(data)


def test_evaluation_reports_misses_and_extras(nlp):
    p = passage("NVIDIA discussed Apple.", synthetic=True)
    result = analyze([p], nlp)
    scores = evaluate_entities(
        result,
        {
            p.passage_id: [
                {"start": 0, "end": 6, "label": "ORG"},
                {"start": 7, "end": 16, "label": "ORG"},
            ]
        },
    )
    assert scores["matched"] == 1
    assert scores["precision"] == scores["recall"] == scores["f1"] == 0.5
    assert len(scores["missed"]) == len(scores["extra"]) == 1
    with pytest.raises(ValueError):
        evaluate_entities(result, {"unknown": []})


def test_unavailable_model_does_not_silently_fallback():
    with pytest.raises(OSError):
        build_pipeline("nonexistent_finsight_model")


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold_sentiment", "bullish"),
        ("gold_reasons", "made_up_category"),
        ("review_status", "revieved"),
        ("review_status", "reviewed"),
    ],
)
def test_invalid_human_annotations_are_reported(nlp, tmp_path, field, value):
    target = tmp_path / "annotations.csv"
    export_annotations(analyze([passage()], nlp), target)
    with target.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0][field] = value
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        annotation_summary(target)
