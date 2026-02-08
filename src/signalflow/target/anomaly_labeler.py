"""Anomaly labeler for black swan event detection in historical data."""

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import sf_component
from signalflow.core.enums import SignalCategory
from signalflow.target.base import Labeler


@dataclass
@sf_component(name="anomaly")
class AnomalyLabeler(Labeler):
    """Labels black swan and flash crash events in historical data.

    Forward-looking labeler that identifies anomalous price movements by
    comparing forward return magnitude against rolling volatility.

    Algorithm:
        1. Compute log returns: log(close[t] / close[t-1])
        2. Compute rolling std of returns over ``vol_window`` bars
        3. Compute forward return magnitude: |log(close[t+horizon] / close[t])|
        4. If forward return > threshold_return_std * rolling_std -> "black_swan"
        5. If additionally the return is negative AND happened in < flash_horizon
           bars -> "flash_crash"
        6. Otherwise -> null (no label)

    Attributes:
        price_col (str): Price column name. Default: "close".
        horizon (int): Forward-looking horizon in bars. Default: 60.
        vol_window (int): Rolling window for volatility estimation. Default: 1440.
        threshold_return_std (float): Number of standard deviations for anomaly
            threshold. Default: 4.0.
        flash_horizon (int): Maximum bars for flash crash classification.
            Default: 10.

    Example:
        ```python
        from signalflow.target.anomaly_labeler import AnomalyLabeler

        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=1440,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        labeled = labeler.compute(ohlcv_df)
        ```

    Note:
        This is a forward-looking labeler -- it uses future data and is NOT
        suitable for live trading. Use ``AnomalyDetector`` for real-time
        anomaly detection.
    """

    signal_category: SignalCategory = SignalCategory.ANOMALY

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
        """Compute anomaly labels for a single pair group.

        Args:
            group_df (pl.DataFrame): Single pair's data sorted by timestamp.
            data_context (dict[str, Any] | None): Additional context.

        Returns:
            pl.DataFrame: Same length as input with anomaly label column added.
                Labels are "black_swan", "flash_crash", or null.
        """
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        if group_df.height == 0:
            return group_df

        price = pl.col(self.price_col)

        # Step 1: log returns
        df = group_df.with_columns(
            (price / price.shift(1)).log().alias("_log_ret"),
        )

        # Step 2: rolling std of returns
        df = df.with_columns(
            pl.col("_log_ret")
            .rolling_std(window_size=self.vol_window, min_samples=max(2, self.vol_window // 4))
            .alias("_rolling_vol"),
        )

        # Step 3: forward return (signed) and magnitude
        df = df.with_columns(
            (price.shift(-self.horizon) / price).log().alias("_forward_ret"),
        )
        df = df.with_columns(
            pl.col("_forward_ret").abs().alias("_forward_ret_abs"),
        )

        # Step 4-5: compute threshold and classify
        # Scale per-bar volatility to horizon-length volatility: vol * sqrt(horizon)
        horizon_threshold = pl.col("_rolling_vol") * self.threshold_return_std * math.sqrt(self.horizon)

        # For flash crash detection, check if a large negative move happens
        # within flash_horizon bars (shorter window).
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
            .then(pl.lit("flash_crash"))
            .when(is_anomaly)
            .then(pl.lit("black_swan"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        # Step 6: meta columns
        if self.include_meta:
            df = df.with_columns(
                [
                    pl.col("_forward_ret").alias("forward_ret"),
                    pl.col("_rolling_vol").alias("vol"),
                ]
            )

        # Clean up temporary columns
        df = df.drop(
            [
                c
                for c in ("_log_ret", "_rolling_vol", "_forward_ret", "_forward_ret_abs", "_flash_ret")
                if c in df.columns
            ]
        )

        # Apply signal masking if configured
        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df
