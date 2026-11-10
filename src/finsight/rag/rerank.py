"""Second-stage reranking: re-score the top hybrid results with a cross-encoder.

Hybrid search ranks paragraphs by comparing the question's vector with each paragraph's
vector, which is fast but rough. A cross-encoder reads the question and one paragraph
together and scores how well that paragraph answers it: slower, so it only re-orders the
top FINSIGHT_RERANK_CANDIDATES (default 30) hybrid results.

    FINSIGHT_RERANK_MODEL   default Xenova/ms-marco-MiniLM-L-6-v2 (~80 MB, fast on CPU);
                            BAAI/bge-reranker-base is stronger but ~1 GB and slower;
                            "overlap" = offline word-overlap scorer for tests/CI only
    FINSIGHT_RERANK_CUDA=1  run it on an NVIDIA GPU (needs fastembed-gpu)

Used by search mode "rerank" in retrieve.py.
"""

from __future__ import annotations

import os
from typing import Protocol

from finsight.rag.embeddings import tokenize

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = int(os.getenv("FINSIGHT_RERANK_CANDIDATES", "30"))


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str = DEFAULT_RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder   # imported here so tests don't need it
        kwargs = {"cuda": True} if os.getenv("FINSIGHT_RERANK_CUDA", "").lower() in ("1", "true", "yes") else {}
        self.name = model
        self._m = TextCrossEncoder(model, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"), **kwargs)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._m.rerank(query, texts, batch_size=32)]


class OverlapReranker:
    """Share of the question's (stemmed) words found in the paragraph. Offline, for tests only."""

    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]


_default: Reranker | None = None


def get_reranker() -> Reranker:
    """One reranker per process (the model loads once)."""
    global _default
    if _default is None:
        model = os.getenv("FINSIGHT_RERANK_MODEL", DEFAULT_RERANK_MODEL)
        _default = OverlapReranker() if model == "overlap" else CrossEncoderReranker(model)
    return _default
