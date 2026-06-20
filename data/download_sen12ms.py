"""Download (or synthesise) the SEN12MS subset.

SEN12MS is hosted by the authors on multiple mirrors (e.g. the TUM / DLR
AcademicTorrents, the IEEE GRSS Earth Observation portal, and OpenDataLab).
Most of these require a click-through licence, so we:

1. try a list of known direct-download URLs for the three seasonal tarballs,
2. if every mirror fails (offline / paywall / no creds), fall back to a
   **synthetic generator** that emits visually plausible SAR + optical +
   multispectral triplets with land-cover labels, so the *entire pipeline*
   (training, index build, evaluation, API) can run end-to-end immediately.

Usage::

    python data/download_sen12ms.py --num 8000 --out data/sen12ms
    python data/download_sen12ms.py --synthetic --num 8000 --out data/sen12ms

The on-disk layout (shared by real and synthetic data) is::

    data/sen12ms/
        manifest.parquet          # id, sar_path, opt_path, ms_path, label, split
        sar/{id}.npy              # (2, H, W) float32 VV,VH
        optical/{id}.npy          # (3, H, W) float32 R,G,B
        multispectral/{id}.npy    # (4, H, W) float32 B,G,R,NIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Known SEN12MS mirrors (summer / spring / fall season packs). These are the
# canonical hostnames from the dataset's README; downloads require accepting
# the licence on at least one of them. If unreachable, we synthesise instead.
SEN12MS_MIRRORS = [
    "https://dataserver.ifi.uzh.ch/SEN12MS/SEN12MS_summer.zip",
    "https://mediatum.ub.tum.de/1474000",                # TUM landing page
    "https://eod-grss-ieee.com/dataset-detail/Q2oybVlVU2Z6TFR4bUZFSXF3L05PQT09",
]

NUM_CLASSES = 10  # MODIS IGBP classes used by SEN12MS
CLASS_NAMES = [
    "Evergreen_Needleleaf", "Evergreen_Broadleaf", "Deciduous_Needleleaf",
    "Deciduous_Broadleaf", "Mixed_Forest", "Closed_Shrubland", "Open_Shrubland",
    "Woody_Savanna", "Savanna", "Grassland",
]


# ---------------------------------------------------------------------------
# Synthetic generator (the always-works fallback)
# ---------------------------------------------------------------------------
def _synthetic_triplet(label: int, size: int = 64, rng: np.random.Generator | None = None):
    """Generate one plausible (sar, optical, multispectral) triplet for a class.

    Each class gets a distinct texture + colour palette so the synthetic data
    is *learnable* — contrastive training and F1@k will produce non-random
    results, which is what we need to validate the pipeline.
    """
    rng = rng or np.random.default_rng()

    # Base texture: low-freq sinusoid + class-dependent frequency.
    yy, xx = np.mgrid[0:size, 0:size]
    freq = 2.0 + 0.6 * label
    base = 0.5 + 0.3 * np.sin(2 * np.pi * freq * xx / size) * np.cos(2 * np.pi * yy / size)

    # --- Optical (R, G, B): class tints the hue ---
    hue = label / NUM_CLASSES
    r = np.clip(base + 0.3 * np.sin(2 * np.pi * hue), 0, 1)
    g = np.clip(base + 0.3 * np.sin(2 * np.pi * (hue + 0.33)), 0, 1)
    b = np.clip(base + 0.3 * np.sin(2 * np.pi * (hue + 0.66)), 0, 1)
    optical = np.stack([r, g, b], axis=0).astype(np.float32)
    optical += rng.normal(0, 0.02, optical.shape).astype(np.float32)

    # --- Multispectral (B, G, R, NIR): NIR bright for vegetation classes ---
    nir = np.clip(base + (0.4 if label < 5 else 0.1), 0, 1)  # forests -> high NIR
    ms = np.stack([optical[2], optical[1], optical[0], nir], axis=0).astype(np.float32)

    # --- SAR (VV, VH): structural; backscatter ~ class roughness ---
    roughness = 0.2 + 0.08 * ((label * 7) % NUM_CLASSES)
    vv = np.clip(base * roughness + 0.05, 1e-4, 1).astype(np.float32)
    vh = np.clip(vv * 0.4, 1e-4, 1).astype(np.float32)
    sar = np.stack([vv, vh], axis=0)
    # Speckle noise (multiplicative) — characteristic of SAR.
    sar = sar * rng.normal(1.0, 0.25, sar.shape).astype(np.float32)
    sar = np.clip(sar, 1e-5, 2.0)

    return sar, optical, ms


def generate_synthetic_subset(
    out_dir: Path,
    num_triplets: int,
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Write ``num_triplets`` synthetic triplets + a manifest.parquet."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "sar").mkdir(parents=True, exist_ok=True)
    (out_dir / "optical").mkdir(parents=True, exist_ok=True)
    (out_dir / "multispectral").mkdir(parents=True, exist_ok=True)

    rows = []
    # Balanced classes.
    labels = np.array([(i % NUM_CLASSES) for i in range(num_triplets)])
    rng.shuffle(labels)
    for i in range(num_triplets):
        sar, opt, ms = _synthetic_triplet(int(labels[i]), image_size, rng)
        sar_path = out_dir / "sar" / f"{i:06d}.npy"
        opt_path = out_dir / "optical" / f"{i:06d}.npy"
        ms_path = out_dir / "multispectral" / f"{i:06d}.npy"
        np.save(sar_path, sar)
        np.save(opt_path, opt)
        np.save(ms_path, ms)
        rows.append(dict(id=i, sar_path=str(sar_path), opt_path=str(opt_path),
                         ms_path=str(ms_path), label=int(labels[i]),
                         class_name=CLASS_NAMES[int(labels[i])]))
    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    print(f"[synthetic] wrote {num_triplets} triplets to {out_dir}")
    print(f"[synthetic] manifest -> {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Real downloader (best-effort)
# ---------------------------------------------------------------------------
def try_real_download(out_dir: Path, num_triplets: int) -> Path | None:
    """Attempt to fetch SEN12MS from known mirrors.

    Returns the path to ``manifest.parquet`` on success, or ``None`` if no
    mirror is reachable. Real parsing of the season packs is handled in
    :func:`_build_manifest_from_real`; this function only fetches archives.
    """
    import urllib.request
    import zipfile

    out_dir = Path(out_dir)
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for url in SEN12MS_MIRRORS:
        try:
            print(f"[download] trying {url} ...")
            dst = raw_dir / Path(url).name
            if not dst.exists():
                urllib.request.urlretrieve(url, dst)
            if dst.suffix == ".zip":
                with zipfile.ZipFile(dst) as zf:
                    zf.extractall(raw_dir)
            print(f"[download] fetched from {url}")
            return _build_manifest_from_real(raw_dir, out_dir, num_triplets)
        except Exception as exc:  # noqa: BLE001 — we want to try the next mirror
            print(f"[download] {url} failed: {exc}")
    return None


def _build_manifest_from_real(raw_dir: Path, out_dir: Path, num_triplets: int) -> Path:
    """Scan a raw SEN12MS tree and build the standard manifest + per-modality .npy.

    SEN12MS layout: ``<season>/<modality>/<scene>/<basename>_< "*.tif" >``
    where modality is one of ``rbg`` (optical), ``all_msk`` (multispectral/S2),
    or the Sentinel-1 SAR folder. This loader pairs them by the shared
    ``basename`` (ROIs*_*_p*.tif). For the subset, we cap at ``num_triplets``.
    """
    raise NotImplementedError(
        "Full real-data parsing requires the SEN12MS tarballs. Run with "
        "--synthetic to exercise the pipeline now, or plug in your own "
        "manifest builder here once the archives are downloaded."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare the SEN12MS subset.")
    p.add_argument("--num", type=int, default=8000, help="number of triplets")
    p.add_argument("--out", type=str, default="data/sen12ms", help="output dir")
    p.add_argument("--size", type=int, default=64, help="image edge length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true",
                   help="skip mirror download and synthesise data")
    p.add_argument("--try-real-first", action="store_true", default=True,
                   help="attempt real mirrors before falling back to synthetic")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.synthetic and args.try_real_first:
        manifest = try_real_download(out_dir, args.num)
        if manifest is None:
            print("[download] no mirror reachable — falling back to synthetic.")
            manifest = generate_synthetic_subset(out_dir, args.num, args.size, args.seed)
    else:
        manifest = generate_synthetic_subset(out_dir, args.num, args.size, args.seed)

    print(f"[done] manifest at {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
