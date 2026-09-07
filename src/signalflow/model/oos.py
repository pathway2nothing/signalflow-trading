"""OOS plumbing - walk-forward fold splitting and the model fingerprint."""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from loguru import logger

from signalflow._hash import stable_hash

__all__ = ["Fold", "build_fingerprint", "make_folds", "rolling_folds", "stable_hash"]


@dataclass
class Fold:
    """One train-before / test-after window of a walk-forward.

    ``train_start`` is ``None`` for an expanding window. ``model`` and ``oos`` are
    filled by :func:`signalflow.model.walkforward.walk_forward`, which keeps the
    per-fold fitted model and its out-of-sample predictions.
    """

    train_end: object
    test_start: object
    test_end: object
    train_start: object = None
    model: object = None
    oos: object = None

    @property
    def tag(self) -> str:
        """``YYYYMM`` of the test window's start - a stable per-fold label (``save_to="..._{tag}"``)."""
        start = self.test_start
        return start.strftime("%Y%m") if hasattr(start, "strftime") else str(start)


def make_folds(ts_unique_sorted: list, n_folds: int) -> list[Fold]:
    """Split sorted unique timestamps into n_folds contiguous blocks."""
    n = len(ts_unique_sorted)
    requested = n_folds
    if n < n_folds + 1:
        n_folds = max(2, min(n_folds, n))
        logger.warning(f"make_folds: requested {requested} folds but only {n} unique timestamps; using {n_folds}")
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    folds: list[Fold] = []
    for k in range(1, n_folds):
        start_i, end_i = bounds[k], bounds[k + 1]
        if end_i <= start_i:
            continue
        folds.append(
            Fold(
                train_end=ts_unique_sorted[start_i - 1],
                test_start=ts_unique_sorted[start_i],
                test_end=ts_unique_sorted[end_i - 1],
            )
        )
    return folds


def rolling_folds(
    ts_unique_sorted: list, refit: timedelta, window: timedelta | None, embargo: timedelta
) -> list[Fold]:
    """Walk-forward folds stepped by ``refit``; each trains on the trailing ``window``.

    Test windows are contiguous ``[test_start, test_start + refit)`` blocks; the
    train span is ``[test_start - embargo - window, test_start - embargo)``, or
    everything before ``test_start - embargo`` when ``window`` is ``None``.
    """
    if not ts_unique_sorted:
        return []
    first, last = ts_unique_sorted[0], ts_unique_sorted[-1]
    folds: list[Fold] = []
    test_start = first + embargo
    while test_start <= last:
        folds.append(
            Fold(
                train_end=test_start - embargo,
                test_start=test_start,
                test_end=test_start + refit,
                train_start=(test_start - embargo - window) if window is not None else None,
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
    tail_cfg: dict | None,
    dataset_params: dict,
    cv: dict,
    output: str,
) -> dict:
    fp = {
        "backend": backend,
        "backend_params": backend_params,
        "target": target_cfg,
        "features": features_cfg,
        "tail": tail_cfg,
        "dataset": dataset_params,
        "cv": cv,
        "output": output,
    }
    fp["model_code"] = stable_hash(
        {"backend": backend, "features": features_cfg, "tail": tail_cfg, "target": target_cfg}
    )
    fp["id"] = stable_hash(fp)
    return fp
