# src/signalflow/detector/macd_cross.py
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import Signals, detector
from signalflow.detector import SignalDetector
from signalflow.feature.base import Feature


@dataclass
class _MacdFeature(Feature):
    """Inline MACD feature for macd_cross detector."""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    price_col: str = "close"

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        fast_ema = pl.col(self.price_col).ewm_mean(span=self.fast_period, adjust=False)
        slow_ema = pl.col(self.price_col).ewm_mean(span=self.slow_period, adjust=False)

        df = df.with_columns(
            (fast_ema - slow_ema).alias("macd_line"),
        )
        df = df.with_columns(
            pl.col("macd_line").ewm_mean(span=self.signal_period, adjust=False).alias("macd_signal"),
        )
        df = df.with_columns(
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist"),
        )
        return df

    @property
    def warmup(self) -> int:
        return self.slow_period * 3


@dataclass
@detector("example/macd_cross")
class ExampleMacdCrossDetector(SignalDetector):
    """MACD signal line crossover detector.

    Signals:
      - "rise": MACD crosses above signal line
      - "fall": MACD crosses below signal line
    """

    allowed_signal_types: set[str] | None = None

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    price_col: str = "close"

    def __post_init__(self) -> None:
        self.allowed_signal_types = {"rise", "fall"}

        self.features = [
            _MacdFeature(
                fast_period=self.fast_period,
                slow_period=self.slow_period,
                signal_period=self.signal_period,
                price_col=self.price_col,
            ),
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        df = features.sort([self.pair_col, self.ts_col])
        df = df.filter(pl.col("macd_line").is_not_null() & pl.col("macd_signal").is_not_null())

        macd = pl.col("macd_line")
        signal = pl.col("macd_signal")
        macd_prev = macd.shift(1).over(self.pair_col)
        signal_prev = signal.shift(1).over(self.pair_col)

        cross_up = (macd > signal) & (macd_prev <= signal_prev)
        cross_down = (macd < signal) & (macd_prev >= signal_prev)

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.when(cross_up)
                .then(pl.lit("rise"))
                .when(cross_down)
                .then(pl.lit("fall"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias("signal_type"),
                pl.when(cross_up).then(1).when(cross_down).then(-1).otherwise(0).alias("signal"),
            ]
        ).filter(pl.col("signal_type").is_not_null())

        return Signals(out)
