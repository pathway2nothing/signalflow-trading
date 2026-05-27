"""Mean-reversion magnitude labeler — regression-style target.

Where :class:`MeanReversionEventLabeler` answers "did the overstretched price
revert at all?" (binary), this labeler answers "how *much* did it revert?".
For every overstretched bar (``|z_now| > stretch_threshold``) we measure the
*best* reduction of ``|z|`` reached within the forward horizon along the
*signed* axis: the path must move toward the mean, not just oscillate around
``z_now``. The continuous ratio drives both a soft three-class target and a
regression meta column, addressing the iter-32 finding that a binary D3
label saturates AUC near 1.0 while still leaving most of the PnL upside on
the table.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.core import labeler
from signalflow.core.enums import SignalCategory
from signalflow.target._soft_helpers import sigmoid_expr
from signalflow.target.base import Labeler

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator if args and callable(args[0]) is False else args[0]

    prange = range  # type: ignore[assignment]


@njit(parallel=True, cache=True)
def _forward_best_revert(
    z_now: np.ndarray,
    price: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """For each bar t with z_now[t] = z0, scan forward up to ``horizon`` bars
    and record the deepest forward z (in the same baseline µ/σ as z0) that lies
    on the path *toward* the mean. If z0 > 0 we track min forward z (clipped
    at 0 from below — we count the move toward 0, not the overshoot through
    the mean). If z0 < 0 we track max forward z (clipped at 0 from above).
    Returns the signed depth ``z_now - z_best`` so larger means stronger
    revert. Bars with no future data, with sd==0, or with |z_now| below
    threshold are filled with NaN by the caller.
    """
    n = price.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in prange(n):
        z0 = z_now[i]
        if np.isnan(z0):
            continue
        sd_i = sd[i]
        mu_i = mu[i]
        if sd_i <= 0 or np.isnan(sd_i) or np.isnan(mu_i):
            continue
        max_j = i + horizon
        if max_j >= n:
            max_j = n - 1
        if z0 > 0:
            best = z0
            for j in range(i + 1, max_j + 1):
                zf = (price[j] - mu_i) / sd_i
                if zf < best:
                    best = zf
                    if best <= 0.0:
                        break
            out[i] = z0 - best  # positive => price moved down toward mean
        elif z0 < 0:
            best = z0
            for j in range(i + 1, max_j + 1):
                zf = (price[j] - mu_i) / sd_i
                if zf > best:
                    best = zf
                    if best >= 0.0:
                        break
            out[i] = best - z0  # positive => price moved up toward mean
        else:
            out[i] = 0.0
    return out


@dataclass
@labeler("mean_reversion_magnitude")
class MeanReversionMagnitudeLabeler(Labeler):
    """Continuous revert-strength target plus three soft buckets.

    Algorithm:
        1. Rolling µ, σ over ``z_window``; ``z_now = (close - µ) / σ``.
        2. For overstretched bars (``|z_now| > stretch_threshold``):
            * scan forward up to ``horizon`` bars,
            * track the deepest forward z in the *toward-mean* direction
              (clipped at 0 — we count progress toward, not through, µ),
            * ``revert_strength = (|z_now| - clip(|z_best|, 0)) / |z_now|``
              ∈ ``[0, 1]``.
        3. Hard label by fixed cuts on ``revert_strength``:
            * ``< partial_threshold``         -> ``"no_revert"``
            * ``[partial_threshold, full_threshold)`` -> ``"partial_revert"``
            * ``>= full_threshold``           -> ``"full_revert"``
        4. Non-overstretched bars get label ``None`` (excluded from training),
           matching the meta-label semantics — we only learn the revert
           magnitude conditional on the price being out of its band.

    Soft outputs (``p_no_revert, p_partial_revert, p_full_revert``) are
    sigmoid-saturated around the two thresholds, so model heads can fit the
    full distribution rather than a hard cut.

    Why this exists (research provenance):
        sf-profit ``experiments_report.md`` 2026-05 — the binary
        ``D3_rev_mean_revert`` reached forward AUC 0.98–0.99 but the resulting
        strategy only captured a small share of available revert size
        (in-sample 50% → OOS 7%). A magnitude target lets downstream models
        size positions by *expected* revert depth instead of a yes/no flag.

    Attributes:
        price_col: Price column. Default ``"close"``.
        horizon: Forward scan window. Default ``240``.
        z_window: Rolling window for µ/σ baseline. Default ``240``.
        stretch_threshold: |z_now| above which a bar is *eligible* and gets
            a label. Default ``2.0``. Bars below are labelled null.
        partial_threshold: revert ratio above which the label is at least
            ``"partial_revert"``. Default ``0.25``.
        full_threshold: revert ratio above which the label is
            ``"full_revert"``. Default ``0.75``.
        softness_k: sigmoid sharpness for the soft three-class output.

    Example:
        ```python
        from signalflow.target import MeanReversionMagnitudeLabeler

        labeler = MeanReversionMagnitudeLabeler(
            horizon=240,
            z_window=240,
            stretch_threshold=2.0,
            partial_threshold=0.25,
            full_threshold=0.75,
            include_meta=True,
            mask_to_signals=False,
        )
        labeled = labeler.compute(ohlcv_df)
        # labeled["revert_strength"] is the continuous regression target.
        ```
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = (
        "no_revert",
        "partial_revert",
        "full_revert",
    )
    softness_k: float = 8.0

    price_col: str = "close"
    horizon: int = 240
    z_window: int = 240
    stretch_threshold: float = 2.0
    partial_threshold: float = 0.25
    full_threshold: float = 0.75

    meta_columns: tuple[str, ...] = ("z_now", "revert_strength")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.z_window <= 1:
            raise ValueError("z_window must be > 1")
        if self.stretch_threshold <= 0:
            raise ValueError("stretch_threshold must be > 0")
        if not (0.0 <= self.partial_threshold < self.full_threshold <= 1.0):
            raise ValueError(
                "Require 0 <= partial_threshold < full_threshold <= 1, "
                f"got partial={self.partial_threshold}, full={self.full_threshold}"
            )

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def _compute_core(self, group_df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """Return (df_with_intermediates, z_now array, revert_strength array)."""
        price = pl.col(self.price_col)
        df = group_df.with_columns(
            price.rolling_mean(self.z_window, min_samples=2).alias("_mu"),
            price.rolling_std(self.z_window, min_samples=2).alias("_sd"),
        ).with_columns(
            pl.when(pl.col("_sd") > 0)
            .then((price - pl.col("_mu")) / pl.col("_sd"))
            .otherwise(pl.lit(None))
            .alias("_z_now"),
        )

        prices = df.get_column(self.price_col).to_numpy().astype(np.float64)
        mu = df.get_column("_mu").fill_null(np.nan).to_numpy().astype(np.float64)
        sd = df.get_column("_sd").fill_null(np.nan).to_numpy().astype(np.float64)
        z_now = df.get_column("_z_now").fill_null(np.nan).to_numpy().astype(np.float64)

        signed_depth = _forward_best_revert(z_now, prices, mu, sd, int(self.horizon))
        abs_z = np.abs(z_now)
        with np.errstate(divide="ignore", invalid="ignore"):
            revert_strength = np.where(abs_z > 0, signed_depth / abs_z, np.nan)
        # Clip into [0, 1] — overshoot through the mean is not "more reverted".
        revert_strength = np.clip(revert_strength, 0.0, 1.0)
        # Mask non-overstretched bars to NaN so they receive null labels.
        revert_strength = np.where(abs_z > self.stretch_threshold, revert_strength, np.nan)
        return df, z_now, revert_strength

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df, z_now, rev = self._compute_core(group_df)
        n = df.height
        labels: list[str | None] = [None] * n
        pt = self.partial_threshold
        ft = self.full_threshold
        for i in range(n):
            r = rev[i]
            if np.isnan(r):
                continue
            if r >= ft:
                labels[i] = "full_revert"
            elif r >= pt:
                labels[i] = "partial_revert"
            else:
                labels[i] = "no_revert"

        df = df.with_columns(pl.Series(self.out_col, labels, dtype=pl.Utf8))

        if self.include_meta:
            df = df.with_columns(
                pl.Series("z_now", z_now, dtype=pl.Float64).fill_nan(None),
                pl.Series("revert_strength", rev, dtype=pl.Float64).fill_nan(None),
            )

        df = df.drop(["_mu", "_sd", "_z_now"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Sigmoid-tercile soft distribution over the revert ratio.

        ``p_full = sigmoid(k * (rev - full_threshold))``;
        ``p_no   = sigmoid(k * (partial_threshold - rev))``;
        ``p_partial = max(1 - p_full - p_no, 0)``, then renormalised. Rows
        where the bar is not overstretched or has no future data emit nulls
        across all three columns.
        """
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        df, _z_now, rev = self._compute_core(group_df)
        df = df.with_columns(pl.Series("_rev", rev, dtype=pl.Float64).fill_nan(None))

        p_full_raw = sigmoid_expr(pl.col("_rev") - pl.lit(self.full_threshold), self.softness_k)
        p_no_raw = sigmoid_expr(pl.lit(self.partial_threshold) - pl.col("_rev"), self.softness_k)
        p_partial_raw = (pl.lit(1.0) - p_full_raw - p_no_raw).clip(lower_bound=0.0)
        total = p_no_raw + p_partial_raw + p_full_raw
        safe = pl.when(total > 0).then(total).otherwise(pl.lit(1.0))
        null_mask = pl.col("_rev").is_null()

        df = df.with_columns(
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_no_raw / safe)
            .alias(f"{self.soft_col_prefix}no_revert"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_partial_raw / safe)
            .alias(f"{self.soft_col_prefix}partial_revert"),
            pl.when(null_mask)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(p_full_raw / safe)
            .alias(f"{self.soft_col_prefix}full_revert"),
        )
        df = df.drop(["_mu", "_sd", "_z_now", "_rev"])
        return df
