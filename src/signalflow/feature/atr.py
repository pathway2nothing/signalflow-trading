from dataclasses import dataclass
from typing import ClassVar, Literal

import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="atr")
class ATRFeature(Feature):
    """Average True Range (ATR) feature.

    Measures market volatility as the moving average of True Range.
    True Range = max(high - low, |high - prev_close|, |low - prev_close|)

    Args:
        period: ATR period. Default: 14.
        smoothing: Smoothing method - "sma" or "ema" (Wilder's). Default: "ema".

    Example:
        >>> atr = ATRFeature(period=14)
        >>> atr.output_cols()  # ["atr_14"]
    """

    period: int = 14
    smoothing: Literal["sma", "ema"] = "ema"

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["atr_{period}"]

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "smoothing": "sma"},
        {"period": 20},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute ATR for a single pair."""
        col_name = self._get_output_name()

        high = pl.col("high")
        low = pl.col("low")
        prev_close = pl.col("close").shift(1)

        # True Range = max(H-L, |H-prevC|, |L-prevC|)
        tr = pl.max_horizontal(
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        )

        # Apply smoothing
        if self.smoothing == "sma":
            atr = tr.rolling_mean(window_size=self.period)
        else:
            # EMA (Wilder's smoothing)
            atr = tr.ewm_mean(span=self.period, adjust=False)

        df = df.with_columns(atr.alias(col_name))

        # Optional z-score normalization
        if self.normalized:
            from signalflow.feature.examples import _get_norm_window, _normalize_zscore

            norm_window = self.norm_period or _get_norm_window(self.period)
            vals = df[col_name].to_numpy()
            normed = _normalize_zscore(vals, window=norm_window)
            df = df.with_columns(pl.Series(name=col_name, values=normed))

        return df

    def _get_output_name(self) -> str:
        suffix = "_norm" if self.normalized else ""
        return f"atr_{self.period}{suffix}"

    @property
    def warmup(self) -> int:
        base = self.period + 1  # +1 for shift in TR calculation
        if self.normalized:
            from signalflow.feature.examples import _get_norm_window

            return base + (self.norm_period or _get_norm_window(self.period))
        return base
