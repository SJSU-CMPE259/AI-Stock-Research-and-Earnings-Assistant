"""Retrieval evaluation for RQ1: sparse (BM25) vs dense (embeddings) vs hybrid.

A case says which paragraphs are relevant to a question, in one of two ways:
  gold_ids    exact passage_ids (best: label them once in the notebook)
  gold_regex  any paragraph whose text matches counts (quick to write, a bit looser)

    cases = [{"id": "Q07", "question": "...", "ticker": "AAPL", "topics": ["risk_factors"],
              "gold_regex": r"tariff|trade restriction"}]
    rows, summary = evaluate(cases, k=5)

Metrics per case and mode: hit@k (any relevant paragraph in the top k), recall@k
(share of gold_ids found; same as hit@k for regex gold) and reciprocal rank.
"""

from __future__ import annotations

import re
from statistics import mean

from finsight.rag.retrieve import latest_accessions, search
from finsight.rag.store import SEARCH_MODES, VectorStore, get_store


def _relevant(case: dict, passage_id: str, text: str) -> bool:
    if case.get("gold_ids"):
        return passage_id in case["gold_ids"]
    return bool(re.search(case["gold_regex"], text, re.IGNORECASE))


def evaluate(cases: list[dict], modes=SEARCH_MODES, k: int = 5, conn=None,
             store: VectorStore | None = None) -> tuple[list[dict], list[dict]]:
    """Run every case through every mode. Returns (per-case rows, per-mode summary)."""
    st = store or get_store()
    rows = []
    for case in cases:
        accs = case.get("accessions")
        if accs is None and case.get("n_filings"):
            accs = latest_accessions(case["ticker"], tuple(case.get("forms", ("10-K", "10-Q"))),
                                     case["n_filings"], conn=conn, store=st)
        for mode in modes:
            hits = search(case["question"], case["ticker"], case.get("topics"), accs,
                          k=k, mode=mode, store=st)
            flags = [_relevant(case, h.passage_id, h.text) for h in hits]
            first = next((i for i, f in enumerate(flags, 1) if f), None)
            if case.get("gold_ids"):
                found = {h.passage_id for h, f in zip(hits, flags) if f}
                recall = len(found) / len(set(case["gold_ids"]))
            else:
                recall = float(first is not None)
            rows.append({"id": case.get("id", case["question"][:30]), "mode": mode,
                         f"hit@{k}": float(first is not None), f"recall@{k}": recall,
                         "rr": 1.0 / first if first else 0.0, "first_hit_rank": first,
                         "top_passage": hits[0].passage_id if hits else None})
    summary = []
    for mode in modes:
        r = [x for x in rows if x["mode"] == mode]
        if r:
            summary.append({"mode": mode, "cases": len(r),
                            f"hit@{k}": round(mean(x[f"hit@{k}"] for x in r), 3),
                            f"recall@{k}": round(mean(x[f"recall@{k}"] for x in r), 3),
                            "MRR": round(mean(x["rr"] for x in r), 3)})
    return rows, summary
