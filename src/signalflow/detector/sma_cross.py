# src/signalflow/detector/sma_cross.py
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import Signals, sf_component
from signalflow.detector import SignalDetector
from signalflow.feature import ExampleSmaFeature


@dataclass
@sf_component(name="example/sma_cross")
class ExampleSmaCrossDetector(SignalDetector):
    """SMA crossover signal detector.

    Signals:
      - "rise": fast crosses above slow
      - "fall": fast crosses below slow
    """

    allowed_signal_types: set[str] | None = None  # set in __post_init__

    fast_period: int = 20
    slow_period: int = 50
    price_col: str = "close"

    def __post_init__(self) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError(f"fast_period must be < slow_period")

        self.fast_col = f"sma_{self.fast_period}"
        self.slow_col = f"sma_{self.slow_period}"
        self.allowed_signal_types = {"rise", "fall"}

        self.features = [
            ExampleSmaFeature(period=self.fast_period, price_col=self.price_col),
            ExampleSmaFeature(period=self.slow_period, price_col=self.price_col),
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        df = features.sort([self.pair_col, self.ts_col])

        df = df.filter(pl.col(self.fast_col).is_not_null() & pl.col(self.slow_col).is_not_null())

        fast = pl.col(self.fast_col)
        slow = pl.col(self.slow_col)
        fast_prev = fast.shift(1).over(self.pair_col)
        slow_prev = slow.shift(1).over(self.pair_col)

        cross_up = (fast > slow) & (fast_prev <= slow_prev)
        cross_down = (fast < slow) & (fast_prev >= slow_prev)

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
