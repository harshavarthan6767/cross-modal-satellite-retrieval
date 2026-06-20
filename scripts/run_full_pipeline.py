"""Run the full pipeline end-to-end (data -> train -> export -> index -> eval).

This is the one-command path that reproduces all results. On a GPU-less laptop
it can run a *tiny* smoke-test config (``--tiny``) using synthetic data so you
can confirm every stage works before doing the real Colab training.

Typical usage:

    # Full run on Colab / a GPU machine:
    python scripts/run_full_pipeline.py --config config/config.yaml

    # Tiny smoke test on the laptop (synthetic data, 2 epochs, ~1 min):
    python scripts/run_full_pipeline.py --tiny

Stages can be skipped with flags so you can, e.g., train on Colab then build
the index + evaluate locally:

    python scripts/run_full_pipeline.py --skip-train --ckpt checkpoints/retrieval.ckpt
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

# Make the repo importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def stage_data(cfg: dict, synthetic: bool) -> None:
    out = Path(cfg["paths"]["data_root"])
    if (out / "manifest.parquet").exists():
        print("[pipeline] data already present, skipping download.")
        return
    cmd = [sys.executable, str(ROOT / "data" / "download_sen12ms.py"),
           "--num", str(cfg["dataset"]["num_triplets"]),
           "--out", str(out),
           "--size", str(cfg["dataset"]["image_size"]),
           "--seed", str(cfg["dataset"]["seed"])]
    if synthetic:
        cmd.append("--synthetic")
    _run(cmd)


def stage_train(cfg: dict, resume: str | None) -> str:
    from training import run_training
    return run_training(cfg, resume=resume)


def stage_export(cfg: dict, ckpt: str) -> None:
    from training.export_onnx import export_all
    export_all(cfg, ckpt)


def stage_index(cfg: dict) -> None:
    from index.build_index import build
    build(cfg)


def stage_eval(cfg: dict, max_queries: int | None) -> None:
    from retrieval.evaluate import evaluate
    evaluate(cfg, max_queries_per_protocol=max_queries)


def tiny_config(cfg: dict) -> dict:
    """A ~1-minute smoke-test config: tiny synthetic data, 2 epochs, CPU."""
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["num_triplets"] = 300
    cfg["train"]["epochs"] = 2
    cfg["train"]["batch_size"] = 64
    cfg["train"]["num_workers"] = 0
    cfg["model"]["pretrained"] = False  # skip the ImageNet download for speed
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the full retrieval pipeline.")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--tiny", action="store_true",
                   help="smoke test: synthetic data, 2 epochs, CPU only")
    p.add_argument("--synthetic", action="store_true",
                   help="use synthetic data (no network download)")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-index", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint to use for export (skips training if set)")
    p.add_argument("--max-queries", type=int, default=None,
                   help="cap queries per protocol during eval")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.tiny:
        cfg = tiny_config(cfg)
        args.synthetic = True
        print("[pipeline] TINY smoke-test mode")

    if not args.skip_data:
        stage_data(cfg, synthetic=args.synthetic)
    if not args.skip_train:
        ckpt = stage_train(cfg, resume=None)
    else:
        ckpt = args.ckpt
        if not ckpt:
            raise SystemExit("--skip-train requires --ckpt PATH")
    if not args.skip_export:
        stage_export(cfg, ckpt)
    if not args.skip_index:
        stage_index(cfg)
    if not args.skip_eval:
        stage_eval(cfg, args.max_queries)

    print("\n[pipeline] all stages complete.")
    print("[pipeline] launch the UI with:  scripts/demo_local.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
