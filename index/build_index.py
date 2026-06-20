"""Build the FAISS gallery index from the trained ONNX embedder.

For every image in the (eval + train) splits and every modality, we:
1. run the matching per-modality ONNX encoder to get a 512-d vector,
2. add it to the FAISS index,
3. record metadata ``{filepath, modality, land_cover_class, label}``.

The resulting ``gallery.index`` + ``gallery_meta.parquet`` are the durable
"encoded store" that the API and evaluator load at query time.

Usage::

    python index/build_index.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from data import SEN12MSDataset
from index.faiss_index import VectorIndex
from retrieval.embedder import ONNXEmbedder, MODALITY_CHANNELS

# Map modality flag to manifest column.
MODALITY_FILE_COL = {
    "sar": "sar_path",
    "optical": "opt_path",
    "multispectral": "ms_path",
}


def _embed_dataset(
    embedder: ONNXEmbedder,
    ds: SEN12MSDataset,
    modality: str,
    batch_size: int = 256,
):
    """Yield (vectors, metadata_rows) for one modality across the dataset."""
    from data.preprocess import augment  # local import to avoid cycles
    col = MODALITY_FILE_COL[modality]
    manifest = ds.manifest.iloc[ds.indices].reset_index(drop=True)
    n = len(manifest)
    all_vecs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Build a batch straight from disk via the dataset's loaders so
        # preprocessing is identical to training/inference.
        batch = np.stack([
            ds._load_modality(manifest.iloc[i][col], modality)
            for i in range(start, end)
        ]).astype(np.float32)
        vecs = embedder.embed(batch, modality)
        all_vecs.append(vecs)
    vectors = np.concatenate(all_vecs, axis=0)
    meta_rows = []
    for i in range(n):
        meta_rows.append({
            "filepath": str(manifest.iloc[i][col]),
            "modality": modality,
            "label": int(manifest.iloc[i]["label"]),
            "class_name": str(manifest.iloc[i].get("class_name", "")),
            "sample_id": int(manifest.iloc[i]["id"]),
        })
    return vectors, meta_rows


def build(cfg: dict | None = None) -> tuple[Path, Path]:
    cfg = cfg or load_config()
    embed_dim = int(cfg["model"]["embed_dim"])
    index_type = cfg["retrieval"]["index_type"]
    onnx_dir = Path(cfg["paths"]["onnx_path"]).parent
    embedder = ONNXEmbedder(onnx_dir, embed_dim=embed_dim)

    # Use the full dataset for the gallery (same- and cross-modal retrieval
    # both need rich, multi-modality coverage).
    ds_all = SEN12MSDataset(
        manifest_path=Path(cfg["paths"]["data_root"]) / "manifest.parquet",
        split="all",
        image_size=cfg["dataset"]["image_size"],
        sar_spec=None, optical_spec=None, ms_spec=None,
    )
    # We need specs for the loader; build from config.
    from config import modality_specs
    specs = modality_specs(cfg)
    ds_all.sar_spec = specs["sar"]
    ds_all.optical_spec = specs["optical"]
    ds_all.ms_spec = specs["multispectral"]

    vidx = VectorIndex.build(dim=embed_dim, index_type=index_type)
    t0 = time.perf_counter()
    total = 0
    for modality in MODALITY_CHANNELS:
        vectors, meta = _embed_dataset(embedder, ds_all, modality,
                                       batch_size=int(cfg["train"]["batch_size"]))
        vidx.add(vectors, meta)
        total += len(vectors)
        print(f"[build] added {len(vectors):6d} {modality:14s} vectors "
              f"(running total {len(vidx)})")
    elapsed = time.perf_counter() - t0

    index_path = Path(cfg["paths"]["gallery_index"])
    meta_path = Path(cfg["paths"]["gallery_meta"])
    vidx.save(index_path, meta_path)
    print(f"[build] {total} vectors indexed in {elapsed:.1f}s "
          f"({1000 * elapsed / max(total,1):.2f} ms/image)")
    print(f"[build] index -> {index_path}")
    print(f"[build] meta  -> {meta_path}")
    return index_path, meta_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the FAISS gallery index.")
    p.add_argument("--config", type=str, default="config/config.yaml")
    args = p.parse_args(argv)
    build(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
