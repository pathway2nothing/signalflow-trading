from dataclasses import dataclass
from typing import ClassVar, Any

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature, GlobalFeature


def _get_norm_window(period: int) -> int:
    """Default normalization window: 3x the feature period, minimum 20."""
    return max(period * 3, 20)


def _normalize_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score normalization for unbounded oscillators."""
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        win = values[i - window + 1 : i + 1]
        valid = win[~np.isnan(win)]
        if len(valid) < 2:
            continue
        mean = np.mean(valid)
        std = np.std(valid, ddof=1)
        if std > 1e-12:
            result[i] = (values[i] - mean) / std
    return result


@dataclass
@sf_component(name="example/rsi")
class ExampleRsiFeature(Feature):
    """Relative Strength Index.

    Bounded oscillator [0, 100]. In normalized mode, rescales to [-1, 1].

    Args:
        period: RSI period. Default: 14.
        price_col: Price column to use. Default: "close".

    Example:
        >>> rsi = ExampleRsiFeature(period=21)
        >>> rsi.output_cols()  # ["rsi_21"]
    """

    period: int = 14
    price_col: str = "close"

    requires = ["{price_col}"]
    outputs = ["rsi_{period}"]

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 21},
    ]

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute RSI for all pairs."""
        return df.group_by(self.group_col, maintain_order=True).map_groups(self.compute_pair)

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute RSI for single pair."""
        col_name = self._get_output_name()

        delta = pl.col(self.price_col).diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)

        avg_gain = gain.rolling_mean(window_size=self.period)
        avg_loss = loss.rolling_mean(window_size=self.period)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df = df.with_columns(rsi.alias(col_name))

        # Normalization: rescale bounded [0, 100] → [-1, 1]
        if self.normalized:
            df = df.with_columns(((pl.col(col_name) - 50) / 50).alias(col_name))

        return df

    def _get_output_name(self) -> str:
        suffix = "_norm" if self.normalized else ""
        return f"rsi_{self.period}{suffix}"

    @property
    def warmup(self) -> int:
        return self.period * 3


@dataclass
@sf_component(name="example/sma")
class ExampleSmaFeature(Feature):
    """Simple Moving Average."""

    period: int = 20
    price_col: str = "close"

    requires = ["{price_col}"]
    outputs = ["sma_{period}"]

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 50},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col_name = self._get_output_name()
        sma = pl.col(self.price_col).rolling_mean(window_size=self.period)
        df = df.with_columns(sma.alias(col_name))

        if self.normalized:
            norm_window = self.norm_period or _get_norm_window(self.period)
            vals = df[col_name].to_numpy()
            normed = _normalize_zscore(vals, window=norm_window)
            df = df.with_columns(pl.Series(name=col_name, values=normed))

        return df

    def _get_output_name(self) -> str:
        suffix = "_norm" if self.normalized else ""
        return f"sma_{self.period}{suffix}"

    @property
    def warmup(self) -> int:
        base = self.period
        if self.normalized:
            return base + (self.norm_period or _get_norm_window(self.period))
        return base


@dataclass
@sf_component(name="example/global_rsi")
class ExampleGlobalMeanRsiFeature(GlobalFeature):
    """Mean RSI across all pairs per timestamp.

    1. Compute RSI per pair
    2. Mean across all pairs at time t -> global_mean_rsi
    3. Optionally: rsi_diff = pair_rsi - global_mean_rsi

    Args:
        period: RSI period. Default: 14.
        add_diff: Add per-pair difference column. Default: False.
    """

    period: int = 14
    price_col: str = "close"
    add_diff: bool = False

    requires = ["{price_col}"]
    outputs = ["global_mean_rsi_{period}"]

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "add_diff": True},
    ]

    def output_cols(self, prefix: str = "") -> list[str]:
        cols = [f"{prefix}global_mean_rsi_{self.period}"]
        if self.add_diff:
            cols.append(f"{prefix}rsi_{self.period}_diff")
        return cols

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        rsi_col = f"rsi_{self.period}"
        out_col = f"global_mean_rsi_{self.period}"

        has_rsi = rsi_col in df.columns
        if not has_rsi:
            rsi = ExampleRsiFeature(period=self.period, price_col=self.price_col)
            df = rsi.compute(df)

        mean_df = df.group_by(self.ts_col).agg(pl.col(rsi_col).mean().alias(out_col))

        df = df.join(mean_df, on=self.ts_col, how="left")

        if self.add_diff:
            df = df.with_columns((pl.col(rsi_col) - pl.col(out_col)).alias(f"rsi_{self.period}_diff"))

        if not has_rsi:
            df = df.drop(rsi_col)

        return df

    @property
    def warmup(self) -> int:
        return self.period * 3
