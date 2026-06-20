"""Central configuration loader.

Every module reads paths and hyperparameters from ``config/config.yaml`` via
:func:`load_config`. Keeping one source of truth avoids drift between the
data pipeline, training, index build, evaluation and API.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


def config_path() -> Path:
    return _CONFIG_PATH


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config as a plain dict.

    ``paths.repo_root`` is resolved to an absolute directory and all relative
    paths in the ``paths`` section are rebased onto it so downstream code can
    use them directly.
    """
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    root = Path(cfg.get("paths", {}).get("repo_root", "."))
    root = root if root.is_absolute() else (_REPO_ROOT / root).resolve()
    cfg.setdefault("paths", {})["repo_root"] = str(root)

    # Rebase the other path entries so callers get absolute paths.
    for key, val in list(cfg["paths"].items()):
        if key == "repo_root":
            continue
        if val is None:
            continue
        vp = Path(val)
        cfg["paths"][key] = str(vp if vp.is_absolute() else (root / vp))
    return cfg


@dataclass
class ModalitySpec:
    """Channel layout + normalisation stats for a modality."""

    name: str
    channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


def modality_specs(cfg: dict[str, Any]) -> dict[str, ModalitySpec]:
    """Return :class:`ModalitySpec` per modality from config."""
    out: dict[str, ModalitySpec] = {}
    for name, spec in cfg.get("modalities", {}).items():
        stats = spec.get("stats", {})
        mean = tuple(stats.get("mean", [0.0] * spec["channels"]))
        std = tuple(stats.get("std", [1.0] * spec["channels"]))
        out[name] = ModalitySpec(name=name, channels=spec["channels"], mean=mean, std=std)
    return out
