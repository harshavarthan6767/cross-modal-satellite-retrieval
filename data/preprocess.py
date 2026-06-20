"""Per-modality preprocessing for SEN12MS (and any uploaded query image).

The same transforms run at train time (with augmentation) and at inference time
(without augmentation), so the gallery and the query live in the same numeric
space. Three modalities are supported:

* ``sar``           — Sentinel-1 VV, VH (2 channels). Converted to decibel
                      scale, clipped, then min-max normalised.
* ``optical``       — Sentinel-2 R, G, B (3 channels).
* ``multispectral`` — Sentinel-2 B, G, R, NIR (4 channels).

Every modality is resized to ``image_size`` (default 64). Normalisation stats
come from ``config.yaml`` so train and inference agree.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Short names used throughout the codebase -> human label.
MODALITIES: tuple[str, ...] = ("sar", "optical", "multispectral")


# ---------------------------------------------------------------------------
# Low-level numeric helpers
# ---------------------------------------------------------------------------
def to_decibel(arr: np.ndarray, clip: tuple[float, float] = (-25.0, 0.0)) -> np.ndarray:
    """Convert linear SAR power to decibel scale and clip.

    ``10 * log10(x)`` is the standard Sentinel-1 conversion. Values are then
    clipped to ``clip`` (default [-25, 0] dB) to suppress extremes.
    """
    arr = np.asarray(arr, dtype=np.float32)
    # Guard against log10(0).
    arr = np.where(arr > 1e-6, arr, 1e-6)
    db = 10.0 * np.log10(arr)
    return np.clip(db, clip[0], clip[1])


def minmax_normalize(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale to [0, 1] per-image using global min/max."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + eps)


def zscore_normalize(arr: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    """Per-channel standardisation: ``(x - mean) / std``.

    ``arr`` shape is ``(C, H, W)``. ``mean``/``std`` have length ``C``.
    """
    arr = np.asarray(arr, dtype=np.float32)
    mean = np.asarray(list(mean), dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(list(std), dtype=np.float32).reshape(-1, 1, 1)
    return (arr - mean) / (std + 1e-6)


def resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a ``(C, H, W)`` array to ``(C, size, size)``.

    Nearest keeps things dependency-free; for 64x64 the quality difference
    vs. bilinear is negligible and we avoid a Pillow round-trip on tensors.
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"expected (C,H,W) array, got shape {arr.shape}")
    c, h, w = arr.shape
    if h == size and w == size:
        return arr.astype(np.float32)
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return arr[:, ys[:, None], xs[None, :]].astype(np.float32)


# ---------------------------------------------------------------------------
# Per-modality pipelines
# ---------------------------------------------------------------------------
def preprocess_sar(
    vv_vh: np.ndarray,
    size: int,
    mean: Iterable[float],
    std: Iterable[float],
    db_clip: tuple[float, float] = (-25.0, 0.0),
) -> np.ndarray:
    """Full SAR pipeline: decibel -> clip -> minmax -> resize -> zscore."""
    if vv_vh.shape[0] != 2:
        raise ValueError("SAR must have 2 channels (VV, VH)")
    db = to_decibel(vv_vh, clip=db_clip)
    norm = minmax_normalize(db)
    resized = resize_nearest(norm, size)
    return zscore_normalize(resized, mean, std)


def preprocess_optical(
    rgb: np.ndarray,
    size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> np.ndarray:
    """Optical (3-ch) pipeline: minmax -> resize -> zscore."""
    if rgb.shape[0] != 3:
        raise ValueError("optical must have 3 channels (R, G, B)")
    norm = minmax_normalize(rgb)
    resized = resize_nearest(norm, size)
    return zscore_normalize(resized, mean, std)


def preprocess_multispectral(
    bgr_nir: np.ndarray,
    size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> np.ndarray:
    """Multispectral (4-ch) pipeline: minmax -> resize -> zscore."""
    if bgr_nir.shape[0] != 4:
        raise ValueError("multispectral must have 4 channels (B, G, R, NIR)")
    norm = minmax_normalize(bgr_nir)
    resized = resize_nearest(norm, size)
    return zscore_normalize(resized, mean, std)


# ---------------------------------------------------------------------------
# Augmentation (train-time only)
# ---------------------------------------------------------------------------
def augment(
    arr: np.ndarray,
    horizontal_flip: bool = True,
    gaussian_noise_std: float = 0.0,
    sar_speckle_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """In-place-friendly augmentation for a ``(C, H, W)`` tensor.

    ``sar_speckle_std`` applies multiplicative speckle noise (Rayleigh-like),
    which is the dominant noise mode of SAR and helps the model generalise.
    """
    rng = rng or np.random.default_rng()
    out = arr.copy()
    if horizontal_flip and rng.random() < 0.5:
        out = out[:, :, ::-1].copy()  # flip W axis
    if gaussian_noise_std > 0:
        out = out + rng.normal(0.0, gaussian_noise_std, size=out.shape).astype(np.float32)
    if sar_speckle_std > 0:
        # Multiplicative speckle: out *= (1 + N), N ~ Normal(0, std)
        speckle = rng.normal(0.0, sar_speckle_std, size=out.shape).astype(np.float32)
        out = out * (1.0 + speckle)
    return out
