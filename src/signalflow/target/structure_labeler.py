"""
Structure labelers (local extrema detection).

Two approaches:

- **StructureLabeler**: Window-based - examines fixed-size windows around each bar.
  Uses Polars expressions for swing/extrema detection.
- **ZigzagStructureLabeler**: Global zigzag - scans the entire price series for
  alternating swing highs/lows that exceed a threshold. Uses sequential algorithm.

Both support fixed-percentage and rolling z-score swing filters.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("structure")
class StructureLabeler(Labeler):
    """
    Label local tops and bottoms using a symmetric window.

    Uses future knowledge (look-forward) to identify bars that are
    local extrema within a combined lookback + lookforward window,
    filtered by either a fixed percentage or a rolling z-score threshold.

    Swing Filter Modes:
        **Fixed percentage** (default): swing must exceed ``min_swing_pct``.

    **Rolling z-score**: set ``min_swing_zscore`` to enable. Computes
        rolling mean and std of window swings over ``vol_window`` bars,
        then filters by z-score >= threshold. Adapts to market volatility
        automatically - tighter in calm markets, wider in volatile ones.

    Algorithm:
        1. For each bar t, examine ``close[t-lookback : t+lookforward+1]``.
        2. Compute swing = ``(window_max - window_min) / window_min``.
        3. If ``close[t]`` is the maximum in that window -> candidate top.
        4. If ``close[t]`` is the minimum in that window -> candidate bottom.
        5. Apply swing filter to confirm:
           - Fixed: ``swing >= min_swing_pct``
           - Z-score: ``(swing - rolling_mean) / rolling_std >= min_swing_zscore``
        6. Otherwise -> ``null``.

    Implementation:
        Uses Polars rolling expressions for computing window max/min and
        detecting extrema, reducing numpy loop overhead.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = ("local_max", "local_min")

    price_col: str = "close"
    lookforward: int = 60
    lookback: int = 60
    min_swing_pct: float = 0.02
    min_swing_zscore: float | None = None
    vol_window: int = 500

    meta_columns: tuple[str, ...] = ("swing_pct",)

    def __post_init__(self) -> None:
        if self.lookforward <= 0:
            raise ValueError("lookforward must be > 0")
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.min_swing_zscore is not None:
            if self.min_swing_zscore <= 0:
                raise ValueError("min_swing_zscore must be > 0")
            if self.vol_window < 20:
                raise ValueError("vol_window must be >= 20 for z-score mode")
        elif self.min_swing_pct < 0:
            raise ValueError("min_swing_pct must be >= 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute structure labels for a single pair."""
        if group_df.height == 0:
            return group_df

        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        price = pl.col(self.price_col)

        lookback_window = self.lookback + 1

        lookback_max = price.rolling_max(window_size=lookback_window, min_samples=1)
        lookback_min = price.rolling_min(window_size=lookback_window, min_samples=1)

        lookforward_max = (
            price.shift(-1).rolling_max(window_size=self.lookforward, min_samples=1).shift(-(self.lookforward - 1))
        )
        lookforward_min = (
            price.shift(-1).rolling_min(window_size=self.lookforward, min_samples=1).shift(-(self.lookforward - 1))
        )

        df = group_df.with_columns(
            [
                lookback_max.alias("_lb_max"),
                lookback_min.alias("_lb_min"),
                lookforward_max.alias("_lf_max"),
                lookforward_min.alias("_lf_min"),
            ]
        )

        df = df.with_columns(
            [
                pl.max_horizontal("_lb_max", "_lf_max").alias("_win_max"),
                pl.min_horizontal("_lb_min", "_lf_min").alias("_win_min"),
            ]
        )

        df = df.with_columns(
            pl.when((pl.col("_win_min") > 0) & (pl.col("_win_max") != pl.col("_win_min")))
            .then((pl.col("_win_max") - pl.col("_win_min")) / pl.col("_win_min"))
            .otherwise(pl.lit(None))
            .alias("_swing")
        )

        df = df.with_columns(
            [
                (price == pl.col("_win_max")).alias("_is_max"),
                (price == pl.col("_win_min")).alias("_is_min"),
            ]
        )

        if self.min_swing_zscore is not None:
            df = df.with_columns(
                [
                    pl.col("_swing").rolling_mean(window_size=self.vol_window, min_samples=20).alias("_swing_mean"),
                    pl.col("_swing").rolling_std(window_size=self.vol_window, min_samples=20).alias("_swing_std"),
                ]
            )

            df = df.with_columns(
                pl.when(pl.col("_swing_std") > 0)
                .then((pl.col("_swing") - pl.col("_swing_mean")) / pl.col("_swing_std"))
                .otherwise(pl.lit(None))
                .alias("_zscore")
            )

            threshold_mask = pl.col("_zscore") >= self.min_swing_zscore
        else:
            threshold_mask = pl.col("_swing") >= self.min_swing_pct

        label_expr = (
            pl.when(threshold_mask & pl.col("_is_max"))
            .then(pl.lit("local_max"))
            .when(threshold_mask & pl.col("_is_min"))
            .then(pl.lit("local_min"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(
                pl.when(pl.col(self.out_col).is_not_null())
                .then(pl.col("_swing"))
                .otherwise(pl.lit(None))
                .alias("swing_pct")
            )

        temp_cols = [
            "_lb_max",
            "_lb_min",
            "_lf_max",
            "_lf_min",
            "_win_max",
            "_win_min",
            "_swing",
            "_is_max",
            "_is_min",
        ]
        if self.min_swing_zscore is not None:
            temp_cols.extend(["_swing_mean", "_swing_std", "_zscore"])

        df = df.drop([c for c in temp_cols if c in df.columns])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df


@dataclass
@register_target("zigzag_structure")
class ZigzagStructureLabeler(Labeler):
    """
    Label local tops and bottoms using a full-series zigzag algorithm.

    Unlike ``StructureLabeler`` (which uses fixed-size windows around each bar),
    this labeler scans the entire price series to find alternating swing
    highs and lows. A new pivot is confirmed only when the price reverses
    by more than the threshold from the current extreme.

    The zigzag algorithm ensures:
        - Tops and bottoms **strictly alternate** (no consecutive tops or bottoms).
        - Each swing exceeds the threshold (either fixed % or adaptive).
        - Pivots are globally consistent across the full series.

    Swing Filter Modes:
        **Fixed percentage** (default): reversal must exceed ``min_swing_pct``.

    **Adaptive (z-score)**: set ``min_swing_zscore`` to enable. Uses
        rolling volatility (std of log-returns) to compute a per-bar
        threshold: ``threshold = zscore x vol x sqrt(vol_window)``.

    Algorithm:
        1. Find first significant swing to determine initial direction.
        2. Track the running extreme (highest high or lowest low).
        3. When price reverses from the extreme by > threshold:
           - Mark the extreme as ``"local_max"`` or ``"local_min"``.
           - Switch direction and start tracking the new extreme.
        4. Result: alternating pivots across the full price series.

    Implementation:
        Uses a sequential state-machine algorithm. This is inherently
        not parallelizable, so numpy/python loops are used. Polars is
        used for rolling volatility computation in z-score mode.
    """

    signal_category: SignalCategory = SignalCategory.PRICE_STRUCTURE

    soft_classes: ClassVar[tuple[str, ...]] = ("local_max", "local_min")

    price_col: str = "close"
    min_swing_pct: float = 0.02
    min_swing_zscore: float | None = None
    vol_window: int = 500

    meta_columns: tuple[str, ...] = ("swing_pct",)

    def __post_init__(self) -> None:
        if self.min_swing_zscore is not None:
            if self.min_swing_zscore <= 0:
                raise ValueError("min_swing_zscore must be > 0")
            if self.vol_window < 20:
                raise ValueError("vol_window must be >= 20 for z-score mode")
        elif self.min_swing_pct <= 0:
            raise ValueError("min_swing_pct must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute zigzag structure labels for a single pair."""
        if group_df.height == 0:
            return group_df

        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        prices = group_df[self.price_col].to_numpy().astype(np.float64)

        if self.min_swing_zscore is not None:
            thresholds = self._adaptive_thresholds(group_df, prices)
        else:
            thresholds = np.full(len(prices), self.min_swing_pct)

        labels, swing_pcts = self._zigzag(prices, thresholds)

        df = group_df.with_columns(pl.Series(name=self.out_col, values=labels, dtype=pl.Utf8))

        if self.include_meta:
            df = df.with_columns(
                pl.Series(
                    name="swing_pct",
                    values=swing_pcts.tolist(),
                    dtype=pl.Float64,
                )
            )

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def _adaptive_thresholds(self, df: pl.DataFrame, prices: np.ndarray) -> np.ndarray:
        """Compute per-bar thresholds: zscore x rolling_vol x sqrt(vol_window)."""
        n = len(prices)
        if n < 2:
            return np.full(n, np.inf)

        price_col = pl.col(self.price_col)
        log_ret = (price_col / price_col.shift(1)).log()

        rolling_vol = log_ret.rolling_std(window_size=self.vol_window, min_samples=20)

        vol_arr = df.select(rolling_vol.alias("vol"))["vol"].to_numpy()

        thresholds = self.min_swing_zscore * vol_arr * np.sqrt(self.vol_window)

        thresholds = np.where(np.isnan(thresholds), np.inf, thresholds)

        return thresholds

    def _zigzag(self, prices: np.ndarray, thresholds: np.ndarray) -> tuple[list[str | None], np.ndarray]:
        """Run zigzag algorithm with per-bar adaptive thresholds."""
        n = len(prices)
        labels: list[str | None] = [None] * n
        swing_pcts = np.full(n, np.nan, dtype=np.float64)

        if n < 2:
            return labels, swing_pcts

        high_idx = 0
        low_idx = 0
        direction = 0
        init_end = 0

        for i in range(1, n):
            if prices[i] > prices[high_idx]:
                high_idx = i
            if prices[i] < prices[low_idx]:
                low_idx = i

            if high_idx != low_idx and prices[low_idx] > 0:
                swing = (prices[high_idx] - prices[low_idx]) / prices[low_idx]
                if swing >= thresholds[i]:
                    if high_idx > low_idx:
                        labels[low_idx] = "local_min"
                        swing_pcts[low_idx] = swing
                        direction = 1
                    else:
                        labels[high_idx] = "local_max"
                        swing_pcts[high_idx] = swing
                        direction = -1
                    init_end = i
                    break

        if direction == 0:
            return labels, swing_pcts

        if direction == 1:
            candidate_idx = high_idx
            candidate_price = prices[high_idx]
        else:
            candidate_idx = low_idx
            candidate_price = prices[low_idx]

        for i in range(init_end + 1, n):
            threshold = thresholds[i]

            if direction == 1:
                if prices[i] >= candidate_price:
                    candidate_idx = i
                    candidate_price = prices[i]
                elif candidate_price > 0:
                    reversal = (candidate_price - prices[i]) / candidate_price
                    if reversal >= threshold:
                        labels[candidate_idx] = "local_max"
                        swing_pcts[candidate_idx] = reversal
                        direction = -1
                        candidate_idx = i
                        candidate_price = prices[i]

            else:
                if prices[i] <= candidate_price:
                    candidate_idx = i
                    candidate_price = prices[i]
                elif candidate_price > 0:
                    reversal = (prices[i] - candidate_price) / candidate_price
                    if reversal >= threshold:
                        labels[candidate_idx] = "local_min"
                        swing_pcts[candidate_idx] = reversal
                        direction = 1
                        candidate_idx = i
                        candidate_price = prices[i]

        return labels, swing_pcts
