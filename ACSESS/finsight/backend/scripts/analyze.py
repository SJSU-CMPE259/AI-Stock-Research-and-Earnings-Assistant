"""Run the standalone NLP baseline on a saved passage JSON file."""

import argparse
import json
from pathlib import Path

from backend.app.agents.analysis import analyze
from backend.app.nlp.annotations import export_annotations, save_result
from backend.app.nlp.entities import build_pipeline
from backend.app.nlp.schemas import Passage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/nlp"))
    parser.add_argument(
        "--model", default="en_core_web_sm", help="Use 'rules' for rules-only NER"
    )
    args = parser.parse_args()
    passages = [
        Passage.model_validate(row) for row in json.loads(args.input.read_text())
    ]
    result = analyze(
        passages, build_pipeline(None if args.model == "rules" else args.model)
    )
    save_result(result, args.output / "analysis.json")
    count = export_annotations(result, args.output / "annotation_queue.csv")
    print(
        f"{len(result.statements)} statements; {count} annotation rows; output: {args.output}"
    )


if __name__ == "__main__":
    main()
