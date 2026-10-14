"""Exact entity-span scores on explicitly annotated examples."""

from .schemas import AnalysisResult


def evaluate_entities(result: AnalysisResult, gold: dict[str, list[dict]]) -> dict:
    """Score only passages present in gold; duplicate spans count once."""
    if not gold or not set(gold) <= result.sources.keys():
        raise ValueError("Gold must name at least one known passage")
    expected = {
        (pid, ent["start"], ent["end"], ent["label"])
        for pid, entities in gold.items()
        for ent in entities
    }
    predicted = {
        (s.passage_id, ent.start, ent.end, ent.label)
        for s in result.statements
        if s.passage_id in gold
        for ent in s.entities
    }
    matched = len(expected & predicted)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(expected) if expected else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "matched": matched,
        "predicted": len(predicted),
        "gold": len(expected),
        "missed": sorted(expected - predicted),
        "extra": sorted(predicted - expected),
    }
