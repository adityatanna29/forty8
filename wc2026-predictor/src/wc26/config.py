"""Config loading + run provenance.

Every entry point calls ``load_config`` and then ``snapshot_run`` so that the
exact configuration and data-manifest hash behind any published number can be
recovered later (ForecastSnapshot provenance, framework §A.4).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and seed all RNGs for reproducibility."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of the resolved config (provenance key)."""
    canon = json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def snapshot_run(cfg: dict[str, Any], manifest_hash: str | None = None) -> Path:
    """Dump resolved config + manifest hash to runs/<timestamp>/ and return the dir."""
    runs_dir = REPO_ROOT / cfg["paths"]["runs"]
    run_dir = runs_dir / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_hash": config_hash(cfg),
        "data_manifest_hash": manifest_hash,
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2))
    return run_dir
