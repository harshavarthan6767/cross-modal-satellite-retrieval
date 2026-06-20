"""ONNX-Runtime image embedder — the CPU inference path for the laptop.

Loads the three per-modality ONNX graphs produced by ``export_onnx.py`` and
exposes a single :meth:`embed` that takes ``(B, C, H, W)`` float32 and returns
``(B, 512)`` L2-normalised vectors. Also offers :meth:`embed_file` and
:meth:`embed_bytes` convenience wrappers used by the API for uploaded images.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image

from config import load_config, modality_specs
from data.preprocess import (
    preprocess_multispectral,
    preprocess_optical,
    preprocess_sar,
)

MODALITY_CHANNELS = {"sar": 2, "optical": 3, "multispectral": 4}
MODALITY_FILE = {
    "sar": "retrieval_model_sar.onnx",
    "optical": "retrieval_model_optical.onnx",
    "multispectral": "retrieval_model_multispectral.onnx",
}


class ONNXEmbedder:
    """Holds one ONNX session per modality + the matching normalisation stats."""

    def __init__(
        self,
        onnx_dir: str | Path,
        embed_dim: int = 512,
        image_size: int = 64,
        providers: list[str] | None = None,
    ) -> None:
        onnx_dir = Path(onnx_dir)
        providers = providers or ["CPUExecutionProvider"]
        self.sessions: dict[str, ort.InferenceSession] = {}
        for modality, fname in MODALITY_FILE.items():
            path = onnx_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"missing ONNX model {path}. Run training/export_onnx.py first."
                )
            self.sessions[modality] = ort.InferenceSession(
                str(path), providers=providers
            )
        self.embed_dim = embed_dim
        self.image_size = image_size
        cfg = load_config()
        self.specs = modality_specs(cfg)

    # -- core ----------------------------------------------------------------
    def embed(self, batch: np.ndarray, modality: str) -> np.ndarray:
        """``(B, C, H, W)`` float32 -> ``(B, embed_dim)`` normalised."""
        if modality not in self.sessions:
            raise KeyError(modality)
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if batch.ndim != 4:
            raise ValueError(f"expected 4D batch, got {batch.shape}")
        sess = self.sessions[modality]
        (out,) = sess.run(None, {"image": batch})
        return out.astype(np.float32)

    # -- from raw uploads ----------------------------------------------------
    def _array_from_image(
        self,
        image: Image.Image,
        modality: str,
    ) -> np.ndarray:
        """Convert a PIL image to the ``(C, H, W)`` array the pipeline expects.

        Uploaded optical/multispectral images are usually 3-channel RGB PNGs;
        we replicate bands to reach the required channel count when the source
        has fewer. SAR uploads are expected as 2-channel (VV, VH) TIFFs/PNGs or
        a single-channel backscatter image that we duplicate to VV=VH.
        """
        spec = self.specs[modality]
        arr = np.asarray(image, dtype=np.float32)
        # Normalise to (H, W, C).
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"unsupported image shape {arr.shape}")
        h, w, c = arr.shape
        # Scale to [0, 1] if it looks like 8-bit / 16-bit.
        if arr.max() > 1.0:
            if arr.max() <= 255:
                arr = arr / 255.0
            elif arr.max() <= 65535:
                arr = arr / 65535.0
        # Match channel count.
        if c < spec.channels:
            # Duplicate the last band to fill (e.g. single SAR band -> VV,VH).
            reps = -(-spec.channels // c)  # ceil div
            arr = np.concatenate([arr] + [arr[..., -1:]] * (reps * c - c), axis=-1) \
                [..., : spec.channels] if False else np.repeat(arr, reps, axis=-1)[..., : spec.channels]
        elif c > spec.channels:
            arr = arr[..., : spec.channels]
        # (H, W, C) -> (C, H, W).
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)

        if modality == "sar":
            arr = preprocess_sar(arr, self.image_size, spec.mean, spec.std)
        elif modality == "optical":
            arr = preprocess_optical(arr, self.image_size, spec.mean, spec.std)
        else:
            arr = preprocess_multispectral(arr, self.image_size, spec.mean, spec.std)
        return arr

    def embed_bytes(self, data: bytes, modality: str) -> np.ndarray:
        """Embed a single uploaded image (PNG/JPEG/TIFF bytes)."""
        img = Image.open(io.BytesIO(data))
        arr = self._array_from_image(img, modality)
        return self.embed(arr[None], modality)[0]

    def embed_file(self, path: str | Path, modality: str) -> np.ndarray:
        with open(path, "rb") as fh:
            return self.embed_bytes(fh.read(), modality)


def build_embedder_from_config(cfg: dict | None = None) -> ONNXEmbedder:
    cfg = cfg or load_config()
    return ONNXEmbedder(
        onnx_dir=Path(cfg["paths"]["onnx_path"]).parent,
        embed_dim=cfg["model"]["embed_dim"],
        image_size=cfg["dataset"]["image_size"],
    )
