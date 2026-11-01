"""Vector DB for filing paragraphs (Qdrant).

One collection, one point per text paragraph:
  vectors  "dense" (embedding) + "bm25" (sparse keywords)
  payload  passage_id, ticker, form, accession, topic, section, seq, text, fiscal labels, url

Where it runs (same code, set by environment):
  FINSIGHT_QDRANT_URL=http://localhost:6333   Qdrant server (Docker / cloud): use this for the
                                              app, for several processes, and for scale
  (unset)                                     embedded, files in data/qdrant/: dev and notebooks;
                                              one process at a time
  FINSIGHT_QDRANT_API_KEY                     for Qdrant Cloud

Run a server:  docker run -p 6333:6333 -v $(pwd)/data/qdrant_server:/qdrant/storage qdrant/qdrant
"""

from __future__ import annotations

import atexit
import os
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from finsight import config
from finsight.rag.embeddings import BM25Encoder, DenseEmbedder, get_dense_embedder

COLLECTION = os.getenv("FINSIGHT_QDRANT_COLLECTION", "finsight_passages")
SEARCH_MODES = ("sparse", "dense", "hybrid")
_NS = uuid.UUID("7f2c1a52-58a4-4d5e-9a57-0e1f6b3d2c11")   # passage_id -> stable point id
_INDEXED_FIELDS = {"ticker": "keyword", "form": "keyword", "accession": "keyword",
                   "topic": "keyword", "splitter_version": "integer", "seq": "integer"}


def point_id(passage_id: str) -> str:
    return str(uuid.uuid5(_NS, passage_id))


@dataclass
class Hit:
    payload: dict
    score: float | None   # higher = better (cosine, BM25 or RRF); None when not ranked


class VectorStore:
    def __init__(self, client: QdrantClient | None = None, dense: DenseEmbedder | None = None,
                 collection: str = COLLECTION):
        self.client = client or default_client()
        self.dense = dense or get_dense_embedder()
        self.sparse = BM25Encoder()
        self.collection = collection
        self._ready = False

    # ---------- setup ----------
    def ensure_collection(self) -> None:
        if self._ready:
            return
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            dim = info.config.params.vectors["dense"].size
            if dim != self.dense.dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has {dim}-dim vectors but the embedder "
                    f"'{self.dense.name}' makes {self.dense.dim}. Rebuild: "
                    f"python scripts/build_rag_index.py --rebuild")
        else:
            self.client.create_collection(
                self.collection,
                vectors_config={"dense": models.VectorParams(size=self.dense.dim,
                                                             distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            for field, kind in _INDEXED_FIELDS.items():   # filters stay fast at scale (server mode)
                try:
                    with warnings.catch_warnings():      # embedded mode ignores them, and says so
                        warnings.simplefilter("ignore")
                        self.client.create_payload_index(self.collection, field, field_schema=kind)
                except Exception:  # noqa: BLE001
                    pass
        self._ready = True

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._ready = False

    # ---------- writes ----------
    def upsert(self, payloads: list[dict], batch: int = 128) -> int:
        """Embed and store paragraphs. Same passage_id -> same point: re-running replaces it."""
        self.ensure_collection()
        for i in range(0, len(payloads), batch):
            chunk = payloads[i:i + batch]
            texts = [_embed_text(p) for p in chunk]
            dense = self.dense.embed_documents(texts)
            sparse = self.sparse.encode_documents(texts)
            self.client.upsert(self.collection, points=[
                models.PointStruct(
                    id=point_id(p["passage_id"]), payload=p,
                    vector={"dense": d, "bm25": models.SparseVector(indices=s[0], values=s[1])})
                for p, d, s in zip(chunk, dense, sparse)])
        return len(payloads)

    def delete_filing(self, accession: str) -> None:
        self.ensure_collection()
        self.client.delete(self.collection, points_selector=models.FilterSelector(
            filter=_filter(accessions=[accession])))

    # ---------- reads ----------
    def count(self, **filters) -> int:
        self.ensure_collection()
        return self.client.count(self.collection, count_filter=_filter(**filters), exact=True).count

    def scroll(self, limit: int = 10_000, **filters) -> list[dict]:
        """All payloads matching the filters (no ranking), in document order."""
        self.ensure_collection()
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, scroll_filter=_filter(**filters),
                                             limit=min(limit, 1000), offset=offset,
                                             with_payload=True, with_vectors=False)
            out += [p.payload for p in pts]
            if offset is None or len(out) >= limit:
                break
        return sorted(out, key=lambda p: (p["filing_date"], p["accession"], p["section_idx"], p["seq"]))

    def search(self, query: str, mode: str = "hybrid", k: int = 8, **filters) -> list[Hit]:
        """Top-k paragraphs: "sparse" (BM25), "dense" (embeddings) or "hybrid" (RRF of both)."""
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        self.ensure_collection()
        flt = _filter(**filters)
        dense = models.Prefetch(query=self.dense.embed_query(query), using="dense",
                                filter=flt, limit=max(k * 4, 40))
        idx, val = self.sparse.encode_query(query)
        sparse = models.Prefetch(query=models.SparseVector(indices=idx, values=val), using="bm25",
                                 filter=flt, limit=max(k * 4, 40))
        if mode == "hybrid":
            res = self.client.query_points(self.collection, prefetch=[dense, sparse],
                                           query=models.FusionQuery(fusion=models.Fusion.RRF),
                                           limit=k, with_payload=True)
        else:
            p = dense if mode == "dense" else sparse
            if mode == "sparse" and not idx:
                return []
            res = self.client.query_points(self.collection, query=p.query, using=p.using,
                                           query_filter=flt, limit=k, with_payload=True)
        return [Hit(pt.payload, pt.score) for pt in res.points]


def _embed_text(p: dict) -> str:
    """What gets embedded: the sub-heading gives a short paragraph its context."""
    return f"{p['heading']}\n{p['text']}" if p.get("heading") else p["text"]


def _filter(ticker: str | None = None, topics: list[str] | None = None,
            accessions: list[str] | None = None, forms: list[str] | None = None,
            splitter_version: int | None = None) -> models.Filter | None:
    must = []
    if ticker:
        must.append(models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper())))
    if topics:
        must.append(models.FieldCondition(key="topic", match=models.MatchAny(any=list(topics))))
    if accessions:
        must.append(models.FieldCondition(key="accession", match=models.MatchAny(any=list(accessions))))
    if forms:
        must.append(models.FieldCondition(key="form", match=models.MatchAny(any=list(forms))))
    if splitter_version is not None:
        must.append(models.FieldCondition(key="splitter_version",
                                          match=models.MatchValue(value=splitter_version)))
    return models.Filter(must=must) if must else None


def default_client() -> QdrantClient:
    url = os.getenv("FINSIGHT_QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("FINSIGHT_QDRANT_API_KEY"))
    path = Path(os.getenv("FINSIGHT_QDRANT_PATH", Path(config.DB_PATH).parent / "qdrant"))
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    atexit.register(client.close)        # release the folder lock cleanly on exit
    return client


_default: VectorStore | None = None


def get_store() -> VectorStore:
    """One shared store per process (the embedding model loads once)."""
    global _default
    if _default is None:
        _default = VectorStore()
    return _default
