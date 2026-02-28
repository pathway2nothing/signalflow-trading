# src/signalflow/detector/rsi_cross.py
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import Signals, detector
from signalflow.detector import SignalDetector
from signalflow.feature import ExampleRsiFeature


@dataclass
@detector("example/rsi_cross")
class ExampleRsiCrossDetector(SignalDetector):
    """RSI oversold/overbought crossover detector.

    Signals:
      - "rise": RSI crosses above oversold level (buy signal)
      - "fall": RSI crosses below overbought level (sell signal)
    """

    allowed_signal_types: set[str] | None = None

    period: int = 14
    oversold: float = 30
    overbought: float = 70
    price_col: str = "close"

    def __post_init__(self) -> None:
        self.rsi_col = f"rsi_{self.period}"
        self.allowed_signal_types = {"rise", "fall"}

        self.features = [
            ExampleRsiFeature(period=self.period, price_col=self.price_col),
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        df = features.sort([self.pair_col, self.ts_col])
        df = df.filter(pl.col(self.rsi_col).is_not_null())

        rsi = pl.col(self.rsi_col)
        rsi_prev = rsi.shift(1).over(self.pair_col)

        # RSI crosses above oversold → buy signal
        cross_up = (rsi > self.oversold) & (rsi_prev <= self.oversold)
        # RSI crosses below overbought → sell signal
        cross_down = (rsi < self.overbought) & (rsi_prev >= self.overbought)

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
