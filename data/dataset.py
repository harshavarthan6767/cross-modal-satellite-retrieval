"""PyTorch Dataset for the (real or synthetic) SEN12MS subset.

Reads ``manifest.parquet`` produced by ``download_sen12ms.py`` and yields, for
each sample, a tuple of the three modalities plus the land-cover label. The
same class is used for training (augmentation on) and for index/eval build
(augmentation off), driven by the ``split`` and ``augment`` arguments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import load_config, modality_specs
from config import ModalitySpec
from data.preprocess import (
    augment,
    preprocess_multispectral,
    preprocess_optical,
    preprocess_sar,
)

Split = Literal["train", "val", "eval", "all"]


class SEN12MSDataset(Dataset):
    """Yields ``(sar, optical, multispectral, label)`` tensors.

    Each modality tensor is ``(C, image_size, image_size)`` float32.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Split = "all",
        split_fractions: tuple[float, float, float] = (0.70, 0.10, 0.20),
        image_size: int = 64,
        sar_spec: ModalitySpec | None = None,
        optical_spec: ModalitySpec | None = None,
        ms_spec: ModalitySpec | None = None,
        augment_cfg: dict | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.manifest = pd.read_parquet(manifest_path)
        self.image_size = image_size
        self.sar_spec = sar_spec
        self.optical_spec = optical_spec
        self.ms_spec = ms_spec
        self.augment_cfg = augment_cfg or {}
        self.seed = seed

        # Slice by split.
        if split == "all":
            self.indices = np.arange(len(self.manifest))
        else:
            from data.splits import make_splits

            fr_train, fr_val, _ = split_fractions
            splits = make_splits(len(self.manifest), fr_train, fr_val, seed)
            self.indices = {"train": splits.train, "val": splits.eval,
                            "eval": splits.eval}[split]
            if split == "train":
                self.indices = splits.train
            elif split == "val":
                self.indices = splits.val
            else:
                self.indices = splits.eval

    def __len__(self) -> int:
        return len(self.indices)

    def _load_modality(self, path: str, kind: str) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if kind == "sar":
            assert self.sar_spec is not None
            return preprocess_sar(arr, self.image_size, self.sar_spec.mean,
                                  self.sar_spec.std)
        if kind == "optical":
            assert self.optical_spec is not None
            return preprocess_optical(arr, self.image_size, self.optical_spec.mean,
                                      self.optical_spec.std)
        if kind == "multispectral":
            assert self.ms_spec is not None
            return preprocess_multispectral(arr, self.image_size, self.ms_spec.mean,
                                            self.ms_spec.std)
        raise ValueError(kind)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[self.indices[idx]]
        sar = self._load_modality(row["sar_path"], "sar")
        opt = self._load_modality(row["opt_path"], "optical")
        ms = self._load_modality(row["ms_path"], "multispectral")

        if self.augment_cfg:
            rng = np.random.default_rng((self.seed, idx))
            flip = bool(self.augment_cfg.get("horizontal_flip", False))
            gn = float(self.augment_cfg.get("gaussian_noise_std", 0.0))
            sp = float(self.augment_cfg.get("sar_speckle_std", 0.0))
            opt = augment(opt, horizontal_flip=flip, gaussian_noise_std=gn, rng=rng)
            ms = augment(ms, horizontal_flip=flip, gaussian_noise_std=gn, rng=rng)
            # SAR gets the extra speckle channel noise.
            sar = augment(sar, horizontal_flip=flip, gaussian_noise_std=gn,
                          sar_speckle_std=sp, rng=rng)

        sar_t = torch.from_numpy(np.ascontiguousarray(sar))
        opt_t = torch.from_numpy(np.ascontiguousarray(opt))
        ms_t = torch.from_numpy(np.ascontiguousarray(ms))
        label = int(row["label"])
        return sar_t, opt_t, ms_t, label


def build_dataset_from_config(
    cfg: dict | None = None,
    split: Split = "train",
    augment: bool = True,
) -> SEN12MSDataset:
    """Convenience factory: build a dataset straight from ``config.yaml``."""
    cfg = cfg or load_config()
    specs = modality_specs(cfg)
    manifest = Path(cfg["paths"]["data_root"]) / "manifest.parquet"
    aug = cfg.get("train", {}).get("aug", {}) if augment else {}
    return SEN12MSDataset(
        manifest_path=manifest,
        split=split,
        split_fractions=(cfg["splits"]["train"], cfg["splits"]["val"], cfg["splits"]["eval"]),
        image_size=cfg["dataset"]["image_size"],
        sar_spec=specs["sar"],
        optical_spec=specs["optical"],
        ms_spec=specs["multispectral"],
        augment_cfg=aug,
        seed=cfg["dataset"]["seed"],
    )
