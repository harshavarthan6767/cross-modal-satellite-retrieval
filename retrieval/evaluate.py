"""Evaluation: F1@5, F1@10 and average retrieval time, per the competition.

Protocols (from ``config.yaml → retrieval.protocols``):

* **same-modal:**  sar2sar, opt2opt, ms2ms
* **cross-modal:** sar2opt, opt2sar, opt2ms, ms2opt

For each query image we embed it with the *source* modality encoder, search the
gallery restricted to the *target* modality, and treat a hit as *relevant* if
it shares the query's land-cover class. F1@k is computed per query then
averaged (macro-F1@k).

Usage::

    python retrieval/evaluate.py --config config/config.yaml

Writes ``results/metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from index.faiss_index import VectorIndex
from retrieval.embedder import ONNXEmbedder, MODALITY_CHANNELS, build_embedder_from_config

MODALITY_FILE_COL = {"sar": "sar_path", "optical": "opt_path", "multispectral": "ms_path"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def f1_at_k(retrieved_labels: np.ndarray, query_label: int, k: int,
            n_relevant_in_gallery: int) -> float:
    """Per-query F1@k.

    ``retrieved_labels``: ordered labels of the top-N hits (N >= k).
    ``n_relevant_in_gallery``: total gallery items sharing the query label
    (within the target modality) — needed for recall.
    """
    if k == 0 or n_relevant_in_gallery == 0:
        return 0.0
    topk = retrieved_labels[:k]
    n_rel = int(np.sum(topk == query_label))
    precision = n_rel / k
    recall = n_rel / n_relevant_in_gallery
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Protocol evaluation
# ---------------------------------------------------------------------------
def _evaluate_protocol(
    embedder: ONNXEmbedder,
    index: VectorIndex,
    manifest: pd.DataFrame,
    protocol: str,
    top_k: int = 10,
    max_queries: int | None = None,
) -> dict:
    """Run one source->target retrieval protocol over the eval split."""
    src, tgt = protocol.split("2")
    src_col = MODALITY_FILE_COL[{"opt": "optical", "ms": "multispectral"}.get(src, src)]
    # Restrict the gallery to the target modality by masking.
    target_mask = (index.meta["modality"] == {"opt": "optical", "ms": "multispectral"}.get(tgt, tgt)).to_numpy()

    # Eval-split queries.
    eval_rows = manifest  # caller may pre-slice
    if max_queries is not None and len(eval_rows) > max_queries:
        rng = np.random.default_rng(0)
        eval_rows = eval_rows.iloc[rng.choice(len(eval_rows), max_queries, replace=False)]

    f1_5, f1_10, times = [], [], []
    for _, row in eval_rows.iterrows():
        # --- embed query (source modality) ---
        from data.dataset import SEN12MSDataset  # reuse loader via a thin shim
        # We load the source array straight from disk + preprocess, matching
        # the dataset pipeline so train/eval agree.
        from config import modality_specs
        from data.preprocess import (
            preprocess_multispectral, preprocess_optical, preprocess_sar,
        )
        specs = modality_specs(load_config())
        arr = np.load(row[src_col]).astype(np.float32)
        src_mod = {"opt": "optical", "ms": "multispectral"}.get(src, src)
        spec = specs[src_mod]
        if src_mod == "sar":
            arr = preprocess_sar(arr, embedder.image_size, spec.mean, spec.std)
        elif src_mod == "optical":
            arr = preprocess_optical(arr, embedder.image_size, spec.mean, spec.std)
        else:
            arr = preprocess_multispectral(arr, embedder.image_size, spec.mean, spec.std)

        t0 = time.perf_counter()
        qvec = embedder.embed(arr[None], src_mod)[0]
        # --- search (then mask to target modality) ---
        # Search a wide pool then keep only target-modality hits.
        scores_full, ids_full = index.index.search(
            qvec.astype(np.float32).reshape(1, -1), min(index.index.ntotal, max(top_k * 4, 50))
        )
        ids = ids_full[0]
        scores = scores_full[0]
        # Filter to target modality, take top_k.
        kept = [(int(i), float(s)) for i, s in zip(ids, scores)
                if i >= 0 and target_mask[i]]
        kept = kept[:top_k]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_labels = np.array(
            [int(index.meta.iloc[i]["label"]) for i, _ in kept], dtype=int
        )
        n_relevant = int(np.sum(
            (index.meta["label"] == row["label"]).to_numpy() & target_mask
        ))
        f1_5.append(f1_at_k(retrieved_labels, int(row["label"]), 5, n_relevant))
        f1_10.append(f1_at_k(retrieved_labels, int(row["label"]), 10, n_relevant))
        times.append(elapsed_ms)

    return {
        "protocol": protocol,
        "n_queries": len(eval_rows),
        "F1@5": float(np.mean(f1_5)) if f1_5 else 0.0,
        "F1@10": float(np.mean(f1_10)) if f1_10 else 0.0,
        "avg_retrieval_time_ms": float(np.mean(times)) if times else 0.0,
    }


def evaluate(cfg: dict | None = None, max_queries_per_protocol: int | None = None) -> dict:
    cfg = cfg or load_config()
    embedder = build_embedder_from_config(cfg)
    index = VectorIndex.load(cfg["paths"]["gallery_index"], cfg["paths"]["gallery_meta"])
    manifest = pd.read_parquet(Path(cfg["paths"]["data_root"]) / "manifest.parquet")

    # Use the eval split only.
    from data.splits import make_splits
    splits = make_splits(len(manifest), cfg["splits"]["train"], cfg["splits"]["val"],
                         seed=cfg["dataset"]["seed"])
    eval_manifest = manifest.iloc[splits.eval].reset_index(drop=True)

    protocols = cfg["retrieval"]["protocols"]
    results = {"same_modal": [], "cross_modal": []}
    for proto in protocols["same_modal"]:
        results["same_modal"].append(
            _evaluate_protocol(embedder, index, eval_manifest, proto,
                               top_k=cfg["retrieval"]["top_k"],
                               max_queries=max_queries_per_protocol)
        )
    for proto in protocols["cross_modal"]:
        results["cross_modal"].append(
            _evaluate_protocol(embedder, index, eval_manifest, proto,
                               top_k=cfg["retrieval"]["top_k"],
                               max_queries=max_queries_per_protocol)
        )

    # Aggregates.
    def agg(rows, metric):
        vals = [r[metric] for r in rows]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "same_modal":  {"F1@5": agg(results["same_modal"], "F1@5"),
                        "F1@10": agg(results["same_modal"], "F1@10"),
                        "avg_retrieval_time_ms": agg(results["same_modal"], "avg_retrieval_time_ms")},
        "cross_modal": {"F1@5": agg(results["cross_modal"], "F1@5"),
                        "F1@10": agg(results["cross_modal"], "F1@10"),
                        "avg_retrieval_time_ms": agg(results["cross_modal"], "avg_retrieval_time_ms")},
    }
    out = {"per_protocol": results, "summary": summary}

    out_path = Path(cfg["paths"]["metrics_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    _print_report(out, out_path)
    return out


def _print_report(out: dict, out_path: Path) -> None:
    s, c = out["summary"]["same_modal"], out["summary"]["cross_modal"]
    print("\n================ Cross-Modal Satellite Retrieval — Report ================")
    print(f"{'Metric':<28}{'Same-modal':>14}{'Cross-modal':>14}")
    print("-" * 56)
    print(f"{'F1@5':<28}{s['F1@5']:>14.4f}{c['F1@5']:>14.4f}")
    print(f"{'F1@10':<28}{s['F1@10']:>14.4f}{c['F1@10']:>14.4f}")
    print(f"{'avg retrieval time (ms)':<28}{s['avg_retrieval_time_ms']:>14.2f}{c['avg_retrieval_time_ms']:>14.2f}")
    print("=" * 56)
    print(f"metrics written -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate retrieval metrics.")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--max-queries", type=int, default=None,
                   help="cap queries per protocol (for quick runs)")
    args = p.parse_args(argv)
    evaluate(load_config(args.config), max_queries_per_protocol=args.max_queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
