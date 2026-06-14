"""OOS plumbing - walk-forward fold splitting and the model fingerprint."""


import hashlib
import json
from dataclasses import dataclass

import numpy as np


def stable_hash(obj) -> str:
    """Deterministic content hash of a JSON-able object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:24]


@dataclass
class Fold:
    train_end_ts: object
    test_start_ts: object
    test_end_ts: object


def median_dt(ts_sorted: list) -> float:
    """Median spacing in seconds between consecutive unique timestamps."""
    if len(ts_sorted) < 2:
        return 0.0
    secs = [(b - a).total_seconds() for a, b in zip(ts_sorted[:-1], ts_sorted[1:], strict=False)]
    secs = [s for s in secs if s > 0]
    return float(np.median(secs)) if secs else 0.0


def make_folds(ts_unique_sorted: list, n_folds: int) -> list[Fold]:
    """Split sorted unique timestamps into n_folds contiguous blocks."""
    n = len(ts_unique_sorted)
    if n < n_folds + 1:
        n_folds = max(2, min(n_folds, n))
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    folds: list[Fold] = []
    for k in range(1, n_folds):
        start_i, end_i = bounds[k], bounds[k + 1]
        if end_i <= start_i:
            continue
        folds.append(
            Fold(
                train_end_ts=ts_unique_sorted[start_i - 1],
                test_start_ts=ts_unique_sorted[start_i],
                test_end_ts=ts_unique_sorted[end_i - 1],
            )
        )
    return folds


def build_fingerprint(
    *,
    backend: str,
    backend_params: dict,
    target_cfg: dict,
    features_cfg: dict,
    encode_cfg: dict | None,
    select_cfg: dict | None,
    dataset_params: dict,
    cv: dict,
    output: str,
) -> dict:
    fp = {
        "backend": backend,
        "backend_params": backend_params,
        "target": target_cfg,
        "features": features_cfg,
        "encode": encode_cfg,
        "select": select_cfg,
        "dataset": dataset_params,
        "cv": cv,
        "output": output,
    }
    fp["model_code"] = stable_hash(
        {"backend": backend, "features": features_cfg, "encode": encode_cfg, "select": select_cfg, "target": target_cfg}
    )
    fp["id"] = stable_hash(fp)
    return fp
