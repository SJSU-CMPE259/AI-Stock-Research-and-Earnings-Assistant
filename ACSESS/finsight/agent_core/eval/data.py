"""Load gold CSVs and make a fixed 50/50 dev/test split."""
from __future__ import annotations

import csv
import random
from pathlib import Path

EVAL_DIR = Path(__file__).parent


def load_csv(name: str) -> list[dict]:
    with open(EVAL_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def dev_test_split(rows: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 split. Tune thresholds on dev, report on test only."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    dev = [rows[i] for i in sorted(idx[:half])]
    test = [rows[i] for i in sorted(idx[half:])]
    return dev, test


def select_split(rows: list[dict], split: str, seed: int = 0) -> list[dict]:
    if split == "all":
        return rows
    dev, test = dev_test_split(rows, seed)
    return dev if split == "dev" else test
