"""Vision Transformer backbones with per-modality input adapters.

The "box-by-box scan" the user described *is* the ViT patch embed stem: it
cuts a 64x64 image into a 4x4 grid of 16x16 patches. Weights are **shared**
across modalities (one ViT for SAR, optical, multispectral), which forces the
network to align the three modalities in a common space — exactly the
cross-modal retrieval objective.

Because the three modalities have different channel counts (2 / 3 / 4), a
light per-modality ``InputAdapter`` projects them to the ViT width before the
shared patch embedding. Adapters are cheap (a single conv) so the parameter
count stays low and CPU inference stays fast.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputAdapter(nn.Module):
    """Project a ``(B, C_in, H, W)`` tensor to ``(B, C_out, H, W)``.

    Implemented as a 1x1 conv so it is a pure per-pixel linear map of channels
    — keeps the patch grid intact and adds negligible compute.
    """

    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SharedViTEncoder(nn.Module):
    """A timm ViT whose patch stem is shared, fed by per-modality adapters.

    Parameters
    ----------
    modality_channels:
        Mapping ``{modality_name: n_input_channels}``, e.g.
        ``{"sar": 2, "optical": 3, "multispectral": 4}``.
    backbone:
        timm model name (default ``vit_small_patch16_224``).
    image_size:
        Spatial size the model actually sees (default 64). The timm ViT is
        re-instantiated at this resolution so positional embeddings match.
    pretrained:
        If True, load ImageNet weights (then fine-tune).
    """

    def __init__(
        self,
        modality_channels: dict[str, int],
        backbone: str = "vit_small_patch16_224",
        image_size: int = 64,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.modality_channels = modality_channels
        self.backbone_name = backbone
        self.image_size = image_size

        # Build the shared ViT at our (small) resolution. patch_size=16,
        # image 64 -> 4x4 = 16 patches per image.
        self.vit = timm.create_model(
            backbone,
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,        # drop the classification head
            global_pool="token",  # use the [CLS] token
        )
        embed_dim = self.vit.num_features
        self.embed_dim = embed_dim

        # Per-modality input adapters. The native ViT stem expects 3 channels
        # (RGB); we adapt each modality to 3 channels before the stem so the
        # ImageNet-pretrained patch embed weights remain meaningful.
        self.adapters = nn.ModuleDict(
            {name: InputAdapter(c, 3) for name, c in modality_channels.items()}
        )

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Return the ``[CLS]`` feature (``B, embed_dim``) for one modality."""
        if modality not in self.adapters:
            raise KeyError(f"unknown modality {modality!r}; "
                           f"known: {list(self.adapters)}")
        adapted = self.adapters[modality](x)
        feat = self.vit(adapted)            # (B, embed_dim) CLS token
        return feat
