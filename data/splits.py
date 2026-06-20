"""Deterministic train/val/eval splits for the SEN12MS subset.

The split is **per location** (per triplet), so the three modalities of one
location always stay together in the same split. Within the eval split, every
sample can act as a query and the rest form the gallery.

Splits are computed from a seed so they are reproducible across the training
notebook, the index builder, and the evaluator — essential for honest metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SplitResult:
    train: np.ndarray
    val: np.ndarray
    eval: np.ndarray

    def __len__(self) -> int:  # total samples
        return len(self.train) + len(self.val) + len(self.eval)


def make_splits(
    n_samples: int,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    seed: int = 42,
) -> SplitResult:
    """Return three index arrays covering ``range(n_samples)``.

    The remainder after train+val goes to eval. Uses a fixed-seed permutation
    so the same ``n_samples``/``seed`` always yields the same split.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    train = np.sort(perm[:n_train])
    val = np.sort(perm[n_train : n_train + n_val])
    ev = np.sort(perm[n_train + n_val :])
    return SplitResult(train=train, val=val, eval=ev)


def stratified_subset(
    labels: Sequence[int],
    n_total: int,
    num_classes: int,
    seed: int = 42,
) -> np.ndarray:
    """Pick a class-balanced subset of indices from a label list.

    Used by the downloader to sample ~``n_total`` triplets spread evenly across
    the ``num_classes`` land-cover classes, which keeps F1@k meaningful (each
    class has enough gallery samples).
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    per_class = max(1, n_total // num_classes)
    chosen: list[int] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    chosen = np.array(sorted(chosen), dtype=int)
    # If classes are imbalanced we may be short; top up randomly.
    if len(chosen) < n_total:
        remaining = np.setdiff1d(np.arange(len(labels)), chosen)
        extra = rng.choice(remaining, size=min(n_total - len(chosen), len(remaining)), replace=False)
        chosen = np.sort(np.concatenate([chosen, extra]))
    return chosen
