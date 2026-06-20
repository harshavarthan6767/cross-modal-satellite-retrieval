"""Contrastive training loop (PyTorch Lightning).

Designed to run on a **Google Colab T4 GPU** (free tier). Checkpoints are
written every epoch so a Colab disconnect never loses more than one epoch.

Run as a script::

    python training/train.py --config config/config.yaml

or import :class:`RetrievalLightningModule` / :func:`run_training` from the
Colab notebook in this folder.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import load_config
from data import build_dataset_from_config
from losses import build_loss_from_config
from models import build_model_from_config


class RetrievalLightningModule(pl.LightningModule):
    """Wraps the model + loss + optimizer for Lightning."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.model = build_model_from_config(cfg)
        self.loss_fn = build_loss_from_config(cfg)
        self.lr = float(cfg["train"]["lr"])
        self.weight_decay = float(cfg["train"]["weight_decay"])

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.model.embed(x, modality)

    def _step(self, batch) -> torch.Tensor:
        sar, opt, ms, _label = batch
        z_sar = self.model.embed(sar, "sar")
        z_opt = self.model.embed(opt, "optical")
        z_ms = self.model.embed(ms, "multispectral")
        return self.loss_fn(z_sar, z_opt, z_ms)

    def training_step(self, batch, _batch_idx):
        loss = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Cosine schedule with linear warmup.
        t_cfg = self.cfg["train"]
        epochs = int(t_cfg["epochs"])
        warmup = int(t_cfg.get("warmup_epochs", 0))
        # trainer.train_dataloader is None at configure-time; use Lightning's
        # own stepping-batch estimate instead.
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * warmup / max(epochs, 1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # +1 so the very first multiplier is non-zero (avoids the
                # "lr_scheduler.step() before optimizer.step()" warning).
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(min(progress, 1.0) * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler,
                                                    "interval": "step"}}


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    tcfg = cfg["train"]
    train_ds = build_dataset_from_config(cfg, split="train", augment=True)
    val_ds = build_dataset_from_config(cfg, split="val", augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=True,
    )
    return train_dl, val_dl


def run_training(cfg: dict | None = None, resume: str | None = None) -> str:
    """Run a full training job; return the path to the best checkpoint."""
    cfg = cfg or load_config()
    pl.seed_everything(int(cfg["dataset"]["seed"]), workers=True)

    train_dl, val_dl = build_dataloaders(cfg)
    module = RetrievalLightningModule(cfg)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        # Note: filename omits the metric value because its name ("val/loss")
        # contains a slash, which Lightning would interpret as a path.
        filename="retrieval-epoch{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=int(cfg["train"].get("checkpoint_every_n_epochs", 1)),
    )
    callbacks = [ckpt_cb, pl.callbacks.LearningRateMonitor(logging_interval="step")]
    if float(cfg["train"].get("grad_clip", 0)) > 0:
        callbacks.append(pl.callbacks.GradientClipCallback(
            gradient_clip_val=float(cfg["train"]["grad_clip"]),
            gradient_clip_algorithm="norm",
        ))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=int(cfg["train"]["epochs"]),
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else 32,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=str(ckpt_dir),
    )
    trainer.fit(module, train_dl, val_dl, ckpt_path=resume)
    best = ckpt_cb.best_model_path
    print(f"[train] best checkpoint: {best}")
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the retrieval model.")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a Lightning checkpoint to resume from")
    args = p.parse_args(argv)
    run_training(load_config(args.config), resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
