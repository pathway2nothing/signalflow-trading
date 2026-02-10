"""Mutual Information estimation for feature-target pairs.

Provides histogram-based MI estimation for continuous and discrete variables.
Used by FeatureInformativenessAnalyzer to measure feature informativeness
against multiple target types.

References:
    - Cover & Thomas (2006) - Elements of Information Theory
    - Kraskov et al. (2004) - MI estimation
"""

from __future__ import annotations

import numpy as np


def entropy_discrete(x: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution.

    H(X) = -sum_x p(x) * log2(p(x))

    Args:
        x: 1D array of discrete values.

    Returns:
        Entropy in bits. NaN if fewer than 2 values.
    """
    x = x[~_isnan_any(x)]
    if len(x) < 2:
        return np.nan

    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def entropy_continuous(x: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy via histogram of a continuous variable.

    Args:
        x: 1D array of continuous values.
        bins: Number of histogram bins.

    Returns:
        Entropy in bits. NaN if fewer than 2 valid values.
    """
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan

    counts, _ = np.histogram(x, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """MI between two discrete (categorical) arrays.

    MI(X;Y) = sum_{x,y} p(x,y) * log2(p(x,y) / (p(x) * p(y)))

    Args:
        x: 1D discrete array.
        y: 1D discrete array of same length.

    Returns:
        MI in bits. NaN if insufficient data.
    """
    mask = ~(_isnan_any(x) | _isnan_any(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan

    return _mi_from_contingency(x, y)


def mutual_information_continuous_discrete(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
) -> float:
    """MI between a continuous feature and a discrete target.

    Bins the continuous variable, then computes MI from the
    joint contingency table of (binned_x, y).

    This is the primary use case: continuous feature columns
    (RSI, SMA, etc.) against discrete labels (RISE/FALL/NONE).

    Args:
        x: 1D continuous feature array.
        y: 1D discrete target array.
        bins: Number of bins for the continuous variable.

    Returns:
        MI in bits. NaN if insufficient data.
    """
    mask = np.isfinite(x) & ~_isnan_any(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan

    x_binned = _bin_continuous(x, bins)
    return _mi_from_contingency(x_binned, y)


def mutual_information_continuous(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
) -> float:
    """MI between two continuous variables.

    Bins both variables and computes MI from the 2D histogram.

    Args:
        x: 1D continuous array.
        y: 1D continuous array.
        bins: Number of bins per dimension.

    Returns:
        MI in bits. NaN if insufficient data.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan

    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    outer = px[:, None] * py[None, :]
    valid = (pxy > 0) & (outer > 0)
    mi = np.sum(pxy[valid] * np.log2(pxy[valid] / outer[valid]))
    return max(mi, 0.0)


def normalized_mutual_information(mi: float, h_x: float, h_y: float) -> float:
    """Normalize MI to [0, 1] using NMI = MI / sqrt(H(X) * H(Y)).

    Args:
        mi: Raw mutual information value.
        h_x: Entropy of X.
        h_y: Entropy of Y.

    Returns:
        NMI in [0, 1]. NaN if either entropy is zero or NaN.
    """
    if np.isnan(mi) or np.isnan(h_x) or np.isnan(h_y):
        return np.nan
    denom = np.sqrt(h_x * h_y)
    if denom <= 0:
        return np.nan
    return min(mi / denom, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _isnan_any(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask for NaN-like values in any dtype."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.isnan(arr)
    if arr.dtype == object:
        return np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in arr])
    return np.zeros(len(arr), dtype=bool)


def _bin_continuous(x: np.ndarray, bins: int) -> np.ndarray:
    """Bin continuous values into integer bin indices."""
    _, edges = np.histogram(x, bins=bins)
    return np.clip(np.digitize(x, edges[:-1]) - 1, 0, bins - 1)


def _mi_from_contingency(x: np.ndarray, y: np.ndarray) -> float:
    """Compute MI from two discrete arrays via contingency table."""
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)

    contingency = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    np.add.at(contingency, (x_idx, y_idx), 1)

    pxy = contingency / contingency.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    outer = px[:, None] * py[None, :]
    valid = (pxy > 0) & (outer > 0)
    mi = np.sum(pxy[valid] * np.log2(pxy[valid] / outer[valid]))
    return max(mi, 0.0)
