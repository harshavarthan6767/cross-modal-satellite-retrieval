"""Projection head that maps ViT features to the shared 512-d embedding space.

A 2-layer MLP with GELU, then optional L2 normalisation. The normalised output
means inner-product similarity in FAISS equals cosine similarity, which is what
the retrieval metrics assume.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 512,
                 hidden_dim: int | None = None,
                 norm_output: bool = True) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm_output = norm_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        if self.norm_output:
            z = nn.functional.normalize(z, dim=-1, eps=1e-6)
        return z
