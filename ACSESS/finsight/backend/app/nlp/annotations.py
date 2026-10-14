"""Export an annotation queue without overwriting human work."""

import csv
import json
from pathlib import Path

from .schemas import AnalysisResult

FIELDS = (
    "statement_id",
    "ticker",
    "text",
    "source_url",
    "accession",
    "form",
    "period_end",
    "section",
    "passage_id",
    "start",
    "end",
    "gold_sentiment",
    "gold_reasons",
    "annotator",
    "review_status",
    "annotation_notes",
)
HUMAN_FIELDS = FIELDS[11:]
LABELS = {"positive", "neutral", "negative", "uncertain", "not_applicable"}
REASONS = {
    "revenue",
    "profitability",
    "earnings",
    "guidance",
    "product_business_growth",
    "market_risk",
    "regulation",
    "competition",
    "investor_analyst_expectations",
}


def annotation_rows(result: AnalysisResult) -> list[dict[str, str]]:
    rows = []
    for statement in result.statements:
        source = result.sources[statement.passage_id]
        if source.source_type == "synthetic":
            continue
        row = {name: "" for name in FIELDS}
        row.update(statement.model_dump(exclude={"entities"}))
        row.update(
            source.model_dump(mode="json", exclude={"text", "ticker", "source_type"})
        )
        row["review_status"] = "unlabeled"
        rows.append(
            {key: "" if value is None else str(value) for key, value in row.items()}
        )
    return rows


def export_annotations(result: AnalysisResult, path: Path) -> int:
    """Merge by stable ID, preserving existing rows and human labels on reruns."""
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("Annotation CSV has an unexpected schema")
            for row in reader:
                if row["statement_id"] in existing:
                    raise ValueError("Duplicate statement ID in annotation CSV")
                existing[row["statement_id"]] = row
    for row in annotation_rows(result):
        previous = existing.get(row["statement_id"])
        if previous:
            if any(previous[key] != row[key] for key in FIELDS[:11]):
                raise ValueError(
                    "Existing evidence changed; refusing to overwrite annotations"
                )
            row.update({key: previous[key] for key in HUMAN_FIELDS})
        existing[row["statement_id"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing.values())
    temporary.replace(path)
    return len(existing)


def annotation_summary(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["gold_sentiment"] and row["gold_sentiment"] not in LABELS:
            raise ValueError(f"Unknown sentiment label: {row['gold_sentiment']}")
        reasons = set(filter(None, row["gold_reasons"].split(";")))
        if not reasons <= REASONS:
            raise ValueError(f"Unknown reason categories: {sorted(reasons - REASONS)}")
        if row["review_status"] not in {"unlabeled", "labeled", "reviewed"}:
            raise ValueError(f"Unknown review status: {row['review_status']}")
        if row["review_status"] in {"labeled", "reviewed"} and not all(
            (row["gold_sentiment"], row["annotator"].strip())
        ):
            raise ValueError("Labeled/reviewed rows require a label and annotator")
    reviewed = [row for row in rows if row["review_status"] == "reviewed"]
    return {
        "total": len(rows),
        "labeled": sum(bool(row["gold_sentiment"]) for row in rows),
        "reviewed": len(reviewed),
        "usable_reviewed": sum(
            row["gold_sentiment"] in {"positive", "neutral", "negative"}
            for row in reviewed
        ),
    }


def save_result(result: AnalysisResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
