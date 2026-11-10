"""Retrieval over filing text, backed by a vector DB (Qdrant) with hybrid search.

    from finsight.rag import retrieve, to_cavm
    result = retrieve("What new risks did Apple flag?", "AAPL")      # search_mode="hybrid" or "rerank"
    evidence = to_cavm(result["evidence"])      # [{"id": "S1", "text": ..., "citation": ...}, ...]

Build the index first: python scripts/build_rag_index.py
"""

from finsight.rag.index import index_all, index_filing
from finsight.rag.retrieve import (RETRIEVAL_MODES, get_section, guess_topics, retrieve, search,
                                   to_cavm)
from finsight.rag.store import SEARCH_MODES, VectorStore, get_store

__all__ = ["index_all", "index_filing", "get_section", "guess_topics", "retrieve", "search",
           "to_cavm", "RETRIEVAL_MODES", "SEARCH_MODES", "VectorStore", "get_store"]
