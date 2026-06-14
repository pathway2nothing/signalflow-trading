"""
Market-wide (cross-sectional) volatility regime labeler.

Unlike :class:`VolatilityRegimeLabeler` (which terciles per-pair forward
vol against its own trailing baseline) this label looks at the
**cross-section** at each timestamp: average forward realised vol over
all pairs present in the input. The resulting market-vol time series is
then terciled against its own trailing distribution - the label is the
*same* value across every pair at a given timestamp.

Use cases:
    * regime conditioning for portfolio strategies;
    * filtering signals during market-wide turbulence;
    * a baseline for cross-asset correlation breaks.
"""


from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._soft_helpers import percentile_tercile_soft
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("market_wide_volatility_regime")
class MarketWideVolatilityRegimeLabeler(Labeler):
    """
    Cross-sectional vol regime: mean forward vol across pairs, terciled.

    Algorithm:
        1. Per pair, compute one-bar log returns and forward realised vol
           over ``horizon`` bars.
        2. At each timestamp, average forward vol across all pairs
           present in the input → market-wide vol series.
        3. Rolling rank percentile over ``lookback_window`` bars of the
           market series.
        4. Broadcast the resulting label back to every (pair, timestamp)
           row so every pair carries the same regime tag.

    Classes (hard): ``mkt_low_vol`` / ``mkt_mid_vol`` / ``mkt_high_vol``.

    Soft API: returns ``[p_mkt_low_vol, p_mkt_mid_vol, p_mkt_high_vol]``
    via the same percentile-tercile sigmoid that
    :class:`VolatilityRegimeLabeler` uses.

    Research provenance:
        iter-33 (sf-profit) ``soft_C1_mkt_vol`` - best soft MI 0.096
        against ``natr_ratio_60_1440`` on the validated pool.
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY

    soft_classes: ClassVar[tuple[str, ...]] = ("mkt_low_vol", "mkt_mid_vol", "mkt_high_vol")
    softness_k: float = 20.0

    price_col: str = "close"
    horizon: int = 60
    upper_quantile: float = 0.67
    lower_quantile: float = 0.33
    lookback_window: int = 1440

    meta_columns: tuple[str, ...] = ("mkt_realized_vol", "mkt_vol_percentile")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not (0.0 < self.lower_quantile < self.upper_quantile < 1.0):
            raise ValueError(
                f"Require 0 < lower_quantile < upper_quantile < 1, got {self.lower_quantile}, {self.upper_quantile}"
            )
        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def _market_series(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return the market-wide volatility series computed once for the whole input."""

        with_ret = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(1).over(self.pair_col)).log().alias("_log_ret")
        )

        with_fwd = with_ret.with_columns(
            pl.col("_log_ret")
            .shift(-1)
            .rolling_std(window_size=self.horizon, min_samples=2)
            .shift(-(self.horizon - 1))
            .over(self.pair_col)
            .alias("_fwd_vol")
        )

        mkt = with_fwd.group_by(self.ts_col).agg(pl.col("_fwd_vol").mean().alias("mkt_realized_vol")).sort(self.ts_col)
        mkt_arr = mkt.get_column("mkt_realized_vol").to_numpy()
        pct = self._rolling_percentile(mkt_arr, self.lookback_window)
        mkt = mkt.with_columns(pl.Series("mkt_vol_percentile", pct, dtype=pl.Float64))

        mkt = mkt.with_columns(
            pl.when(pl.col("mkt_vol_percentile").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("mkt_vol_percentile") > self.upper_quantile)
            .then(pl.lit("mkt_high_vol"))
            .when(pl.col("mkt_vol_percentile") < self.lower_quantile)
            .then(pl.lit("mkt_low_vol"))
            .otherwise(pl.lit("mkt_mid_vol"))
            .alias(self.out_col)
        )

        p_low, p_mid, p_high = percentile_tercile_soft(
            pl.col("mkt_vol_percentile"),
            lower_q=self.lower_quantile,
            upper_q=self.upper_quantile,
            k=self.softness_k,
        )
        mkt = mkt.with_columns(
            p_low.alias(f"{self.soft_col_prefix}mkt_low_vol"),
            p_mid.alias(f"{self.soft_col_prefix}mkt_mid_vol"),
            p_high.alias(f"{self.soft_col_prefix}mkt_high_vol"),
        )
        return mkt

    @staticmethod
    def _rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            v = values[i]
            if v is None or np.isnan(v):
                continue
            start = max(0, i - window + 1)
            win = values[start : i + 1]
            valid = win[~np.isnan(win)]
            if len(valid) < 20:
                continue
            out[i] = float(np.mean(valid <= v))
        return out

    def compute(
        self,
        df: pl.DataFrame,
        signals: Any | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Override base ``compute`` - needs full cross-section, not per-pair group_by."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute expects pl.DataFrame, got {type(df)}")
        self._validate_input_pl(df)
        mkt = self._market_series(df)

        keep_cols = [self.out_col]
        if self.include_meta:
            keep_cols += list(self.meta_columns)
        out = df.sort([self.pair_col, self.ts_col]).join(
            mkt.select([self.ts_col, *keep_cols]), on=self.ts_col, how="left"
        )
        if self.keep_input_columns:
            return out
        return out.select([self.pair_col, self.ts_col, *keep_cols])

    def compute_soft(
        self,
        df: pl.DataFrame,
        signals: Any | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Override base ``compute_soft`` - same cross-section logic."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute_soft expects pl.DataFrame, got {type(df)}")
        self._validate_input_pl(df)
        mkt = self._market_series(df)
        soft_cols = [f"{self.soft_col_prefix}{c}" for c in self.soft_classes]
        out = df.sort([self.pair_col, self.ts_col]).join(
            mkt.select([self.ts_col, *soft_cols]), on=self.ts_col, how="left"
        )
        if self.keep_input_columns:
            return out
        return out.select([self.pair_col, self.ts_col, *soft_cols])

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """Not used - :meth:`compute` is overridden for cross-sectional aggregation."""
        raise NotImplementedError(f"{self.__class__.__name__} is cross-sectional; use compute() directly.")
