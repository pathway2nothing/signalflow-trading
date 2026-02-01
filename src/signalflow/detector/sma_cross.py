# src/signalflow/detector/sma_cross.py
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import Signals, SignalType, sf_component
from signalflow.detector import SignalDetector
from signalflow.feature import FeaturePipeline, SmaFeature


@dataclass
@sf_component(name="example/sma_cross")
class ExampleSmaCrossDetector(SignalDetector):
    """SMA crossover signal detector.

    Signals:
      - RISE: fast crosses above slow
      - FALL: fast crosses below slow
    """

    fast_period: int = 20
    slow_period: int = 50
    price_col: str = "close"

    def __post_init__(self) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError(f"fast_period must be < slow_period")

        self.fast_col = f"sma_{self.fast_period}"
        self.slow_col = f"sma_{self.slow_period}"

        self.feature_pipeline = FeaturePipeline(features=[
            SmaFeature(period=self.fast_period, price_col=self.price_col),
            SmaFeature(period=self.slow_period, price_col=self.price_col),
        ])

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        df = features.sort([self.pair_col, self.ts_col])

        df = df.filter(
            pl.col(self.fast_col).is_not_null() & 
            pl.col(self.slow_col).is_not_null()
        )

        fast = pl.col(self.fast_col)
        slow = pl.col(self.slow_col)
        fast_prev = fast.shift(1).over(self.pair_col)
        slow_prev = slow.shift(1).over(self.pair_col)

        cross_up = (fast > slow) & (fast_prev <= slow_prev)
        cross_down = (fast < slow) & (fast_prev >= slow_prev)

        out = df.select([
            self.pair_col,
            self.ts_col,
            pl.when(cross_up).then(pl.lit(SignalType.RISE.value))
              .when(cross_down).then(pl.lit(SignalType.FALL.value))
              .otherwise(pl.lit(SignalType.NONE.value))
              .alias("signal_type"),
            pl.when(cross_up).then(1)
              .when(cross_down).then(-1)
              .otherwise(0)
              .alias("signal"),
        ])

        return Signals(out)