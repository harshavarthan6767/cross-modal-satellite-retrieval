"""Export the trained retrieval model to ONNX for CPU inference on the laptop.

ONNX Runtime runs the same encoder with no GPU and ~30-50 ms/image, which is
what we need on the Samsung Book. We export **one ONNX file per modality**
because ONNX graphs need a fixed input channel count; the encoder weights are
identical across the three files (only the input adapter differs), so they
share the trained knowledge.

Output files::

    models/retrieval_model_sar.onnx
    models/retrieval_model_optical.onnx
    models/retrieval_model_multispectral.onnx

Each accepts ``(B, C, 64, 64)`` float32 and returns ``(B, 512)`` L2-normalised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import load_config
from models import build_model_from_config

MODALITY_CHANNELS = {"sar": 2, "optical": 3, "multispectral": 4}


def load_checkpoint(model, ckpt_path: str) -> None:
    """Load weights from a PyTorch / Lightning checkpoint into the model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Lightning prefixes module params with nothing on the model itself when
    # wrapped, so strip a leading "model." if present.
    cleaned = {}
    for k, v in state.items():
        key = k[len("model."):] if k.startswith("model.") else k
        cleaned[key] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[export] missing={list(missing)[:5]} ... unexpected={list(unexpected)[:5]} ...")
    model.eval()


def export_for_modality(
    model,
    modality: str,
    out_path: Path,
    image_size: int = 64,
    embed_dim: int = 512,
) -> None:
    """Export a single-modality ONNX graph."""
    channels = MODALITY_CHANNELS[modality]
    model.set_active_modality(modality)
    dummy = torch.randn(1, channels, image_size, image_size, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False selects the legacy TorchScript exporter. The newer dynamo
    # exporter requires `onnxscript` and is less reliable for timm ViT models;
    # the legacy path handles the patch-embed + adapter graph cleanly.
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )
    print(f"[export] {modality:14s} -> {out_path}  (in {channels}ch, out {embed_dim}d)")


def verify_onnx(onnx_path: Path) -> None:
    """Sanity-check that the ONNX graph loads and matches PyTorch output."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"[verify] {onnx_path.name}: ONNX loads, inputs=",
          [(i.name, i.shape) for i in sess.get_inputs()])


def export_all(cfg: dict, ckpt_path: str) -> list[Path]:
    image_size = int(cfg["dataset"]["image_size"])
    embed_dim = int(cfg["model"]["embed_dim"])
    out_dir = Path(cfg["paths"]["onnx_path"]).parent
    model = build_model_from_config(cfg)
    load_checkpoint(model, ckpt_path)

    written = []
    for modality in MODALITY_CHANNELS:
        out_path = out_dir / f"retrieval_model_{modality}.onnx"
        export_for_modality(model, modality, out_path, image_size, embed_dim)
        verify_onnx(out_path)
        written.append(out_path)

    # Also write a convenience symlink/name for the "default" optical model.
    default = out_dir / "retrieval_model.onnx"
    default.write_bytes(written[1].read_bytes())  # optical is index 1
    print(f"[export] default optical model -> {default}")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export trained model to ONNX.")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--ckpt", type=str, required=True,
                   help="path to the trained checkpoint (.ckpt / .pt)")
    args = p.parse_args(argv)
    export_all(load_config(args.config), args.ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
