"""Losses package."""
from losses.contrastive import InfoNCELoss, build_loss_from_config  # noqa: F401

__all__ = ["InfoNCELoss", "build_loss_from_config"]
