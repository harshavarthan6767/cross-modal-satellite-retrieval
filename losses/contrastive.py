"""InfoNCE / NT-Xent contrastive loss over the three modalities.

For each location ``i`` in a batch we have three views of the *same* scene:
``z_sar[i], z_opt[i], z_ms[i]`` (the user's "assign the 3 modes as the same
image"). For every ordered pair of modalities ``(a, b)`` we treat ``z_a[i]`` as
an anchor, ``z_b[i]`` as the positive, and every other ``z_b[j != i]`` as a
negative, then apply the symmetric InfoNCE objective. Averaging over all six
ordered pairs gives a strong cross-modal alignment signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE across the three modalities.

    Inputs must already be L2-normalised (the model's head does this).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def _nce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """One-way InfoNCE: anchor modality -> target modality.

        ``anchor``, ``positive``: ``(B, D)`` L2-normalised.
        """
        # Similarity matrix (B, B); diagonal = positive pairs.
        logits = anchor @ positive.t() / self.temperature
        targets = torch.arange(anchor.size(0), device=anchor.device)
        # Cross-entropy with the diagonal as the correct class.
        loss = F.cross_entropy(logits, targets)
        return loss

    def forward(
        self,
        z_sar: torch.Tensor,
        z_opt: torch.Tensor,
        z_ms: torch.Tensor,
    ) -> torch.Tensor:
        """Average InfoNCE over all 6 ordered modality pairs."""
        views = {"sar": z_sar, "optical": z_opt, "multispectral": z_ms}
        names = list(views)
        total = 0.0
        n_pairs = 0
        for a in names:
            for b in names:
                if a == b:
                    # Same-modality: still useful as a regulariser (augmented
                    # views of the same image should match), but skip to keep
                    # the focus on cross-modal alignment.
                    continue
                total = total + self._nce(views[a], views[b])
                n_pairs += 1
        return total / max(n_pairs, 1)


def build_loss_from_config(cfg: dict) -> InfoNCELoss:
    l = cfg["loss"]
    return InfoNCELoss(temperature=l["temperature"])
