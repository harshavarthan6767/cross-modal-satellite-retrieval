"""Pydantic request/response schemas for the retrieval API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["sar", "optical", "multispectral"]


class HealthResponse(BaseModel):
    status: str = "ok"
    gallery_size: int
    embed_dim: int


class IndexResponse(BaseModel):
    """Result of adding an uploaded image to the gallery."""

    added: int
    gallery_size: int
    modality: Modality
    label: int | None = None


class HitItem(BaseModel):
    rank: int
    score: float = Field(..., description="cosine similarity, [-1, 1]")
    modality: Modality
    label: int
    class_name: str = ""
    thumbnail_b64: str | None = Field(
        None, description="base64 PNG thumbnail of the gallery image"
    )
    filepath: str


class QueryResponse(BaseModel):
    """Ranked top-k retrieval result for one query crop."""

    modality: Modality
    top_5: list[HitItem]
    top_10: list[HitItem]
    embed_ms: float
    search_ms: float
    total_ms: float
    gallery_size: int
