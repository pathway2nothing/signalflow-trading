"""Deterministic resampling statistics for scorecards."""


import numpy as np


def _as_array(returns) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def bootstrap_ci(returns, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the *mean* return."""
    arr = _as_array(returns)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def monte_carlo_bounds(returns, n: int = 1000, horizon: int | None = None, seed: int = 0) -> dict:
    """
    Resample per-period returns into ``n`` synthetic equity paths and report
    the terminal-equity distribution.

    Each path is a sequence of ``horizon`` returns drawn with replacement from
    the observed returns (``horizon`` defaults to the number of observed
    returns); terminal equity is the cumulative product starting from 1.0.
    Returns 5th/50th/95th percentiles plus mean/min/max. Deterministic for a
    fixed ``seed``.
    """
    arr = _as_array(returns)
    h = int(horizon) if horizon is not None else arr.size
    if arr.size == 0 or h <= 0:
        return {
            "p5": 1.0, "p50": 1.0, "p95": 1.0,
            "mean": 1.0, "min": 1.0, "max": 1.0,
            "horizon": h, "n_paths": int(n),
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, h))
    paths = arr[idx]
    terminal = np.prod(1.0 + paths, axis=1)
    return {
        "p5": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "mean": float(terminal.mean()),
        "min": float(terminal.min()),
        "max": float(terminal.max()),
        "horizon": h,
        "n_paths": int(n),
    }
