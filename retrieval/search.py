"""Query-time retrieval: embed the query crop, search FAISS, return ranked hits.

This is the user's "upload a cropped image, scan box-by-box, find a match"
path. We also measure per-query wall-clock time, which is one of the
competition metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from index.faiss_index import SearchHit, VectorIndex
from retrieval.embedder import ONNXEmbedder


@dataclass
class QueryResult:
    """A single query's ranked output."""

    hits: list[SearchHit]
    embed_ms: float
    search_ms: float

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms


class Retriever:
    """Glue between the ONNX embedder and the FAISS index."""

    def __init__(self, embedder: ONNXEmbedder, index: VectorIndex,
                 top_k: int = 10) -> None:
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def query_vector(self, vector: np.ndarray, top_k: int | None = None) -> tuple[list[SearchHit], float]:
        """Search a pre-computed query vector; return hits + search time (ms)."""
        k = top_k or self.top_k
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        t0 = time.perf_counter()
        (hits,) = self.index.search(vector, k)
        return hits, (time.perf_counter() - t0) * 1000.0

    def query_bytes(self, data: bytes, modality: str, top_k: int | None = None) -> QueryResult:
        """Full path for an uploaded query crop."""
        t0 = time.perf_counter()
        vec = self.embedder.embed_bytes(data, modality)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        hits, search_ms = self.query_vector(vec, top_k)
        return QueryResult(hits=hits, embed_ms=embed_ms, search_ms=search_ms)


def build_retriever_from_config(cfg: dict | None = None) -> Retriever:
    from config import load_config
    from retrieval.embedder import build_embedder_from_config

    cfg = cfg or load_config()
    embedder = build_embedder_from_config(cfg)
    index = VectorIndex.load(cfg["paths"]["gallery_index"], cfg["paths"]["gallery_meta"])
    return Retriever(embedder, index, top_k=cfg["retrieval"]["top_k"])
