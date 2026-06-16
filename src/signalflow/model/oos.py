"""OOS plumbing - walk-forward fold splitting and the model fingerprint."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def stable_hash(obj) -> str:
    """Deterministic content hash of a JSON-able object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:24]


def parse_duration(s: str) -> timedelta:
    """Parse a duration like ``1d``, ``365d``, ``12h``, ``30m`` into a timedelta."""
    s = s.strip().lower()
    return timedelta(seconds=float(s[:-1]) * _UNIT_SECONDS[s[-1]])


@dataclass
class Fold:
    train_end_ts: object
    test_start_ts: object
    test_end_ts: object
    train_start_ts: object = None


def median_dt(ts_sorted: list) -> float:
    """Median spacing in seconds between consecutive unique timestamps."""
    if len(ts_sorted) < 2:
        return 0.0
    secs = [(b - a).total_seconds() for a, b in itertools.pairwise(ts_sorted)]
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


def rolling_folds(ts_unique_sorted: list, refit: timedelta, window: timedelta, embargo: timedelta) -> list[Fold]:
    """Walk-forward folds stepped by ``refit``; each trains on the trailing ``window``.

    Test windows are contiguous ``[test_start, test_start + refit)`` blocks; the
    train span is ``[test_start - embargo - window, test_start - embargo)``.
    """
    if not ts_unique_sorted:
        return []
    first, last = ts_unique_sorted[0], ts_unique_sorted[-1]
    folds: list[Fold] = []
    test_start = first + embargo
    while test_start <= last:
        folds.append(
            Fold(
                train_end_ts=test_start - embargo,
                test_start_ts=test_start,
                test_end_ts=test_start + refit,
                train_start_ts=test_start - embargo - window,
            )
        )
        test_start = test_start + refit
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
