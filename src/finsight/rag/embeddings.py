"""Text -> vectors for the vector DB.

Two kinds of vector per paragraph, so one collection supports all three RQ1 setups:
  dense   meaning (embedding model): "chip sales limits" finds "export restrictions"
  sparse  keywords (BM25 weights): exact names like "Epic Games", "Item 1A"

    get_dense_embedder()   chosen by FINSIGHT_EMBED_MODEL (default BAAI/bge-small-en-v1.5)
    BM25Encoder()          sparse vectors; Qdrant applies IDF across the collection

FINSIGHT_EMBED_MODEL=hash selects HashEmbedder: no download, deterministic,
bag-of-words only. Use it for unit tests and CI, never for real results.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import zlib
from collections import Counter
from typing import Iterable, Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, fast on CPU, strong for its size

STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because
been before being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in into is it
its itself just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[&'.-][a-z0-9]+)*")


class DenseEmbedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """ONNX embedding model via fastembed: CPU by default, no PyTorch needed.

    The model is downloaded once (~70 MB for bge-small) into FASTEMBED_CACHE_PATH
    or fastembed's default cache.
    """

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 64):
        from fastembed import TextEmbedding   # imported here so tests don't need it
        self.name = model
        self.batch_size = batch_size
        self._m = TextEmbedding(model, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        self.dim = len(next(iter(self._m.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._m.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._m.query_embed(text))).tolist()


class HashEmbedder:
    """Hashing-trick bag of words. Offline and deterministic; knows no synonyms."""

    def __init__(self, dim: int = 256):
        self.name, self.dim = f"hash-{dim}", dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in tokenize(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
 
# Factory: picks which embedder to use.
# Checks the `model` arg, then env var FINSIGHT_EMBED_MODEL, then DEFAULT_MODEL.
#   "hash" / "hash-512" -> HashEmbedder (fast, offline, for tests)
#   anything else       -> FastEmbedEmbedder (real model)
# Always query an index with the same embedder that built it.
def get_dense_embedder(model: str | None = None) -> DenseEmbedder:
    model = model or os.getenv("FINSIGHT_EMBED_MODEL", DEFAULT_MODEL)
    if model == "hash" or model.startswith("hash-"):
        return HashEmbedder(int(model.split("-")[1]) if "-" in model else 256)
    return FastEmbedEmbedder(model)


# ---------- sparse (BM25) ----------

# Stemmer: cuts words to their root so "acquisitions" matches "acquisition".
# Uses py_rust_stemmers if installed (comes with fastembed), else no stemming.
# Index and queries must both stem or both not, or keyword search breaks.
def _stemmer():
    try:   # installed with fastembed; falls back to no stemming
        from py_rust_stemmers import SnowballStemmer
        return SnowballStemmer("english").stem_word
    except Exception:  # noqa: BLE001
        return lambda w: w


_STEM = _stemmer()

# Split text into clean words: lowercase, drop filler words, shorten to roots.
# "The revenues increased" -> ["revenu", "increas"]
def tokenize(text: str) -> list[str]:
    return [_STEM(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


# Turn a word into a fixed number ID (same word = same ID, every time).
def _index(token: str) -> int:
    return zlib.crc32(token.encode())   # stable across runs and machines


class BM25Encoder:
    """BM25 term weights as a sparse vector. The collection uses Qdrant's IDF
    modifier, so document frequency is applied at query time over the whole index
    (no fitting step, nothing to rebuild when filings are added)."""

    name = "bm25-v1"

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 220.0):
        self.k1, self.b, self.avg_len = k1, b, avg_len   # avg_len ~ paragraph size from sections.py

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        tf = Counter(_index(t) for t in toks)
        norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
        idx = sorted(tf)
        return idx, [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in idx]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        idx = sorted({_index(t) for t in tokenize(text)})
        return idx, [1.0] * len(idx)

    def encode_documents(self, texts: Iterable[str]) -> list[tuple[list[int], list[float]]]:
        return [self.encode_document(t) for t in texts]
