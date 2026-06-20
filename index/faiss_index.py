"""FAISS vector index — the project's "encoded numbers/binaries" store.

Each gallery image becomes one L2-normalised embedding; we store all of them
in a FAISS ``IndexFlatIP`` (inner product = cosine after normalisation) plus a
sidecar metadata table (``id -> {filepath, modality, land_cover_class}``). The
table is saved as Parquet — the durable binary record the user described.

For the full 180k SEN12MS dataset, swap ``flat_ip`` for ``ivf_pq`` in
``config.yaml`` to get sub-linear search (see :class:`VectorIndex`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


@dataclass
class SearchHit:
    """One retrieved gallery item."""

    id: int
    score: float
    metadata: dict[str, Any]


@dataclass
class VectorIndex:
    """Thin wrapper around a FAISS index + metadata sidecar.

    Vectors must be L2-normalised (the embedding head guarantees this) so inner
    product == cosine similarity.
    """

    dim: int
    index: faiss.Index
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_type: str = "flat_ip"

    # ---- construction ------------------------------------------------------
    @classmethod
    def build(cls, dim: int, index_type: str = "flat_ip",
              ivf_nlist: int = 256, pq_m: int = 32, pq_nbits: int = 8) -> "VectorIndex":
        if index_type == "flat_ip":
            idx = faiss.IndexFlatIP(dim)
        elif index_type == "ivf_pq":
            quantizer = faiss.IndexFlatIP(dim)
            idx = faiss.IndexIVFPQ(quantizer, dim, ivf_nlist, pq_m, pq_nbits)
            idx.nprobe = max(8, ivf_nlist // 8)
        else:
            raise ValueError(f"unknown index_type {index_type!r}")
        return cls(dim=dim, index=idx, index_type=index_type)

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> "VectorIndex":
        idx = faiss.read_index(str(index_path))
        meta = pd.read_parquet(meta_path)
        return cls(dim=idx.d, index=idx, meta=meta,
                   index_type="ivf_pq" if isinstance(idx, faiss.IndexIVFPQ)
                   else "flat_ip")

    # ---- mutation ----------------------------------------------------------
    def add(self, vectors: np.ndarray, meta_rows: list[dict[str, Any]]) -> None:
        """Add a batch of normalised vectors + matching metadata rows."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if self.index_type == "ivf_pq" and not self.index.is_trained:
            # Need enough points to train; fall back to flat if not.
            self.index.train(vectors)
        start = self.index.ntotal
        self.index.add(vectors)
        new_rows = pd.DataFrame(meta_rows)
        new_rows["id"] = np.arange(start, start + len(vectors))
        self.meta = pd.concat([self.meta, new_rows], ignore_index=True)

    def save(self, index_path: str | Path, meta_path: str | Path) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        self.meta.to_parquet(meta_path, index=False)

    # ---- search ------------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int = 10) -> list[list[SearchHit]]:
        """Return ``len(query)`` lists of :class:`SearchHit` ranked by score."""
        query = np.ascontiguousarray(query, dtype=np.float32)
        scores, ids = self.index.search(query, top_k)
        results: list[list[SearchHit]] = []
        for row in range(query.shape[0]):
            hits = []
            for rank in range(top_k):
                gid = int(ids[row, rank])
                if gid < 0:
                    continue
                md = self.meta.iloc[gid].to_dict()
                hits.append(SearchHit(id=gid, score=float(scores[row, rank]), metadata=md))
            results.append(hits)
        return results

    def __len__(self) -> int:
        return int(self.index.ntotal)
