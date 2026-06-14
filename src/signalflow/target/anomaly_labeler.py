"""Anomaly labeler for black swan event detection in historical data."""

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target._soft_helpers import signed_tercile_soft
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("anomaly")
class AnomalyLabeler(Labeler):
    """
    Labels black swan and flash crash events in historical data.

    Forward-looking labeler that identifies anomalous price movements by
    comparing forward return magnitude against rolling volatility.

    Algorithm:
        1. Compute log returns: log(close[t] / close[t-1])
        2. Compute rolling std of returns over ``vol_window`` bars
        3. Compute forward return magnitude: |log(close[t+horizon] / close[t])|
        4. If forward return > threshold_return_std * rolling_std -> "extreme_positive_anomaly"
        5. If additionally the return is negative AND happened in < flash_horizon
           bars -> "extreme_negative_anomaly"
        6. Otherwise -> null (no label)
    """

    signal_category: SignalCategory = SignalCategory.ANOMALY

    soft_classes: ClassVar[tuple[str, ...]] = (
        "extreme_negative_anomaly",
        "normal",
        "extreme_positive_anomaly",
    )

    price_col: str = "close"
    horizon: int = 60
    vol_window: int = 1440
    threshold_return_std: float = 4.0
    flash_horizon: int = 10

    meta_columns: tuple[str, ...] = ("forward_ret", "vol")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.vol_window <= 0:
            raise ValueError("vol_window must be > 0")
        if self.threshold_return_std <= 0:
            raise ValueError("threshold_return_std must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute anomaly labels for a single pair group."""
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        if group_df.height == 0:
            return group_df

        price = pl.col(self.price_col)


        df = group_df.with_columns(
            (price / price.shift(1)).log().alias("_log_ret"),
        )


        df = df.with_columns(
            pl.col("_log_ret")
            .rolling_std(window_size=self.vol_window, min_samples=max(2, self.vol_window // 4))
            .alias("_rolling_vol"),
        )


        df = df.with_columns(
            (price.shift(-self.horizon) / price).log().alias("_forward_ret"),
        )
        df = df.with_columns(
            pl.col("_forward_ret").abs().alias("_forward_ret_abs"),
        )


        horizon_threshold = pl.col("_rolling_vol") * self.threshold_return_std * math.sqrt(self.horizon)


        flash_threshold = pl.col("_rolling_vol") * self.threshold_return_std * math.sqrt(self.flash_horizon)
        df = df.with_columns(
            (price.shift(-self.flash_horizon) / price).log().alias("_flash_ret"),
        )

        is_anomaly = (
            pl.col("_forward_ret_abs").is_not_null()
            & pl.col("_rolling_vol").is_not_null()
            & (pl.col("_forward_ret_abs") > horizon_threshold)
        )

        is_flash_crash = (
            is_anomaly
            & pl.col("_flash_ret").is_not_null()
            & (pl.col("_flash_ret") < 0)
            & (pl.col("_flash_ret").abs() > flash_threshold)
        )

        label_expr = (
            pl.when(is_flash_crash)
            .then(pl.lit("extreme_negative_anomaly"))
            .when(is_anomaly)
            .then(pl.lit("extreme_positive_anomaly"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)


        if self.include_meta:
            df = df.with_columns(
                [
                    pl.col("_forward_ret").alias("forward_ret"),
                    pl.col("_rolling_vol").alias("vol"),
                ]
            )


        df = df.drop(
            [
                c
                for c in ("_log_ret", "_rolling_vol", "_forward_ret", "_forward_ret_abs", "_flash_ret")
                if c in df.columns
            ]
        )


        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft triple ``(p_extreme_negative_anomaly, p_normal, p_extreme_positive_anomaly)``."""
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if group_df.height == 0:
            return group_df

        price = pl.col(self.price_col)
        df = group_df.with_columns((price / price.shift(1)).log().alias("_log_ret"))
        df = df.with_columns(
            pl.col("_log_ret")
            .rolling_std(window_size=self.vol_window, min_samples=max(2, self.vol_window // 4))
            .alias("_rolling_vol"),
        )
        df = df.with_columns((price.shift(-self.horizon) / price).log().alias("_forward_ret"))

        scale = math.sqrt(self.horizon)
        z = (
            pl.when(pl.col("_rolling_vol") > 0)
            .then(pl.col("_forward_ret") / (pl.col("_rolling_vol") * scale))
            .otherwise(pl.lit(None))
        )
        df = df.with_columns(z.alias("_z"))

        p_neg, p_norm, p_pos = signed_tercile_soft(
            pl.col("_z"),
            neg_threshold=self.threshold_return_std,
            pos_threshold=self.threshold_return_std,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_neg.alias(f"{self.soft_col_prefix}extreme_negative_anomaly"),
            p_norm.alias(f"{self.soft_col_prefix}normal"),
            p_pos.alias(f"{self.soft_col_prefix}extreme_positive_anomaly"),
        )
        df = df.drop([c for c in ("_log_ret", "_rolling_vol", "_forward_ret", "_z") if c in df.columns])
        return df
