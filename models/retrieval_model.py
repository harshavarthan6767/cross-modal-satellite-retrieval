"""Full retrieval model: shared ViT encoder + embedding head.

The model exposes:

* :meth:`forward_triplet` — used at training time; returns three normalised
  embeddings (one per modality) for a batch of co-registered triplets. The
  contrastive loss (``losses.contrastive``) aligns these.
* :meth:`embed` — used at index/query time; embeds a single modality batch.
  This is also the symbol exported to ONNX (via a thin wrapper) so the same
  encoder runs on the CPU laptop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.backbones import SharedViTEncoder
from models.embedding_head import EmbeddingHead

MODALITY_CHANNELS = {"sar": 2, "optical": 3, "multispectral": 4}


class RetrievalModel(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        image_size: int = 64,
        embed_dim: int = 512,
        pretrained: bool = True,
        norm_output: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SharedViTEncoder(
            modality_channels=MODALITY_CHANNELS,
            backbone=backbone,
            image_size=image_size,
            pretrained=pretrained,
        )
        self.head = EmbeddingHead(
            in_dim=self.encoder.embed_dim,
            embed_dim=embed_dim,
            norm_output=norm_output,
        )
        self.embed_dim = embed_dim

    def embed(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Embed a single modality batch -> (B, embed_dim), L2-normalised."""
        feat = self.encoder(x, modality)
        return self.head(feat)

    def forward_triplet(
        self, sar: torch.Tensor, optical: torch.Tensor, multispectral: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_sar, z_opt, z_ms), each (B, embed_dim) and normalised."""
        z_sar = self.embed(sar, "sar")
        z_opt = self.embed(optical, "optical")
        z_ms = self.embed(multispectral, "multispectral")
        return z_sar, z_opt, z_ms

    # -- single-modality forward for ONNX export -----------------------------
    # ONNX needs a fixed input signature; we export one modality at a time by
    # instantiating a small wrapper (see training/export_onnx.py) rather than
    # baking the modality string into the graph. Here we just provide a
    # generic forward that dispatches on a module-level "active modality".
    _active_modality: str = "optical"

    def set_active_modality(self, modality: str) -> None:
        if modality not in MODALITY_CHANNELS:
            raise ValueError(modality)
        type(self)._active_modality = modality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x, type(self)._active_modality)


def build_model_from_config(cfg: dict) -> RetrievalModel:
    """Construct a :class:`RetrievalModel` from the config dict."""
    m = cfg["model"]
    return RetrievalModel(
        backbone=m["backbone"],
        image_size=cfg["dataset"]["image_size"],
        embed_dim=m["embed_dim"],
        pretrained=m["pretrained"],
        norm_output=m["norm_output"],
    )
