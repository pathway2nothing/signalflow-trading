"""Soft-label helpers — sigmoid / percentile / Gaussian probability calibration.

Used by :class:`signalflow.target.Labeler` subclasses to convert continuous
metrics (forward returns, percentiles, z-scores, Hurst exponents) into
calibrated probability distributions instead of hard threshold cuts.

Conventions:
    * All builders return a tuple of Polars expressions, one per class, in the
      order that matches the subclass's :attr:`soft_classes`.
    * Probability rows sum to 1.0 over the returned expressions; renormalisation
      is applied where saturation could otherwise break the sum.
    * Where any input expression is null the returned probabilities are null
      so callers can mask invalid bars uniformly.
"""
from __future__ import annotations

import polars as pl


def sigmoid_expr(x: pl.Expr, k: float) -> pl.Expr:
    """``1 / (1 + exp(-k * x))`` as a Polars expression."""
    return pl.lit(1.0) / (pl.lit(1.0) + (-k * x).exp())


def gaussian_expr(x: pl.Expr, center: float, k: float) -> pl.Expr:
    """Unnormalised Gaussian membership ``exp(-0.5 * (k * (x - center)) ** 2)``."""
    z = k * (x - pl.lit(center))
    return (-0.5 * z * z).exp()


def binary_threshold_soft(
    metric: pl.Expr,
    threshold: float,
    k: float,
) -> tuple[pl.Expr, pl.Expr]:
    """Soft binary classification: ``(p_below, p_above)``.

    ``p_above = sigmoid(k * (metric - threshold))``;
    ``p_below = 1 - p_above``. Both are null where ``metric`` is null.
    """
    p_above = sigmoid_expr(metric - pl.lit(threshold), k)
    null_mask = metric.is_null()
    p_above = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_above)
    p_below = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.lit(1.0) - p_above)
    return p_below, p_above


def signed_tercile_soft(
    metric: pl.Expr,
    neg_threshold: float,
    pos_threshold: float,
    k: float,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Soft three-class classification of a signed metric: ``(p_neg, p_mid, p_pos)``.

    ``p_pos = sigmoid(k * (metric - pos_threshold))``,
    ``p_neg = sigmoid(k * (-metric - |neg_threshold|))``,
    ``p_mid = max(1 - p_pos - p_neg, 0)``.

    The triple is then renormalised so the row sums to 1.0 even when both tails
    saturate (which would otherwise push ``p_mid`` below zero).
    """
    pos_thr = abs(pos_threshold)
    neg_thr = abs(neg_threshold)
    p_pos_raw = sigmoid_expr(metric - pl.lit(pos_thr), k)
    p_neg_raw = sigmoid_expr(-metric - pl.lit(neg_thr), k)
    p_mid_raw = (pl.lit(1.0) - p_pos_raw - p_neg_raw).clip(lower_bound=0.0)
    total = p_neg_raw + p_mid_raw + p_pos_raw
    safe = pl.when(total > 0).then(total).otherwise(pl.lit(1.0))
    null_mask = metric.is_null()
    p_neg = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_neg_raw / safe)
    p_mid = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_mid_raw / safe)
    p_pos = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_pos_raw / safe)
    return p_neg, p_mid, p_pos


def percentile_tercile_soft(
    percentile: pl.Expr,
    lower_q: float,
    upper_q: float,
    k: float,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Soft tercile from a percentile column (0..1): ``(p_low, p_mid, p_high)``.

    ``p_high = sigmoid(k * (pct - upper_q))``,
    ``p_low  = sigmoid(k * (lower_q - pct))``,
    ``p_mid  = max(1 - p_high - p_low, 0)``,
    then the row is renormalised. ``k`` is interpreted in *percentile units*
    — for default ``upper_q=0.67, lower_q=0.33`` a ``k`` of ~20 gives a sharp
    cut at the boundary while still spreading mass into the middle bucket.
    """
    p_high_raw = sigmoid_expr(percentile - pl.lit(upper_q), k)
    p_low_raw = sigmoid_expr(pl.lit(lower_q) - percentile, k)
    p_mid_raw = (pl.lit(1.0) - p_high_raw - p_low_raw).clip(lower_bound=0.0)
    total = p_low_raw + p_mid_raw + p_high_raw
    safe = pl.when(total > 0).then(total).otherwise(pl.lit(1.0))
    null_mask = percentile.is_null()
    p_low = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_low_raw / safe)
    p_mid = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_mid_raw / safe)
    p_high = pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(p_high_raw / safe)
    return p_low, p_mid, p_high


def gaussian_membership_soft(
    metric: pl.Expr,
    centers: tuple[float, ...],
    k: float,
) -> tuple[pl.Expr, ...]:
    """Multi-class Gaussian membership: one probability per ``center``.

    For each center ``c`` we compute ``exp(-0.5 * (k * (metric - c)) ** 2)``,
    then renormalise so the per-row total is 1. The same null-mask propagation
    as the other helpers applies.
    """
    raws = [gaussian_expr(metric, c, k) for c in centers]
    total = raws[0]
    for r in raws[1:]:
        total = total + r
    safe = pl.when(total > 0).then(total).otherwise(pl.lit(1.0))
    null_mask = metric.is_null()
    return tuple(
        pl.when(null_mask).then(pl.lit(None, dtype=pl.Float64)).otherwise(r / safe)
        for r in raws
    )
