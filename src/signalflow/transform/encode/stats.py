"""WoE / IV statistics and binning primitives."""


import numpy as np

__all__ = ["quantile_edges", "monotonic_edges", "assign_bins", "compute_woe_table", "information_value"]


def quantile_edges(x: np.ndarray, max_bins: int) -> np.ndarray:
    """Internal bin boundaries from unique quantiles of ``x`` (NaNs ignored)."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    edges = np.unique(np.quantile(finite, qs))
    return edges


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values to bin indices in ``[0, len(edges)]``; NaN → -1 (missing)."""
    out = np.full(x.shape, -1, dtype=np.int64)
    mask = np.isfinite(x)
    if edges.size == 0:
        out[mask] = 0
        return out
    out[mask] = np.searchsorted(edges, x[mask], side="right")
    return out


def compute_woe_table(
    bins: np.ndarray, y: np.ndarray, n_bins: int, alpha: float = 0.5
) -> tuple[np.ndarray, float]:
    """Return (woe per bin index, IV) for a binned feature."""
    pos_total = float(np.sum(y == 1))
    neg_total = float(np.sum(y == 0))
    pos_total = max(pos_total, 1e-9)
    neg_total = max(neg_total, 1e-9)

    woe = np.zeros(n_bins + 1, dtype=float)
    iv = 0.0
    for b in range(n_bins + 1):
        idx = b if b < n_bins else -1
        sel = bins == idx
        pos_b = float(np.sum(y[sel] == 1)) if sel.any() else 0.0
        neg_b = float(np.sum(y[sel] == 0)) if sel.any() else 0.0
        dist_pos = (pos_b + alpha) / (pos_total + alpha * (n_bins + 1))
        dist_neg = (neg_b + alpha) / (neg_total + alpha * (n_bins + 1))
        w = float(np.log(dist_pos / dist_neg))
        woe[b] = w
        iv += (dist_pos - dist_neg) * w
    return woe, float(iv)


def information_value(bins: np.ndarray, y: np.ndarray, n_bins: int, alpha: float = 0.5) -> float:
    """IV of a binned feature."""
    return compute_woe_table(bins, y, n_bins, alpha)[1]


def monotonic_edges(x: np.ndarray, y: np.ndarray, max_bins: int) -> np.ndarray:
    """Quantile bins merged until the per-bin event rate is monotonic."""
    edges = quantile_edges(x, max_bins)
    if edges.size == 0:
        return edges
    finite_mask = np.isfinite(x)
    xf, yf = x[finite_mask], y[finite_mask]

    def rates(e: np.ndarray) -> np.ndarray:
        b = assign_bins(xf, e)
        n = e.size + 1
        return np.array([yf[b == i].mean() if np.any(b == i) else np.nan for i in range(n)])

    while edges.size > 0:
        r = rates(edges)
        r = r[~np.isnan(r)]
        if r.size <= 2:
            break
        inc = np.all(np.diff(r) >= -1e-12)
        dec = np.all(np.diff(r) <= 1e-12)
        if inc or dec:
            break

        diffs = np.diff(r)

        trend = np.sign(r[-1] - r[0]) or 1.0
        violation = -diffs * trend
        worst = int(np.argmax(violation))
        edges = np.delete(edges, min(worst, edges.size - 1))
    return edges
