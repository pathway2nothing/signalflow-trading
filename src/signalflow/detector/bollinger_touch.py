# src/signalflow/detector/bollinger_touch.py
from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import Signals, detector
from signalflow.detector import SignalDetector
from signalflow.feature.base import Feature


@dataclass
class _BollingerFeature(Feature):
    """Inline Bollinger Bands feature for bollinger_touch detector."""

    period: int = 20
    std_dev: float = 2.0
    price_col: str = "close"

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        middle = pl.col(self.price_col).rolling_mean(window_size=self.period)
        std = pl.col(self.price_col).rolling_std(window_size=self.period)

        df = df.with_columns(
            middle.alias("bb_middle"),
            (middle + self.std_dev * std).alias("bb_upper"),
            (middle - self.std_dev * std).alias("bb_lower"),
        )
        return df

    @property
    def warmup(self) -> int:
        return self.period * 2


@dataclass
@detector("example/bollinger_touch")
class ExampleBollingerTouchDetector(SignalDetector):
    """Bollinger Bands touch/cross detector.

    Signals:
      - "rise": Price crosses below lower band (buy signal / mean reversion)
      - "fall": Price crosses above upper band (sell signal / mean reversion)
    """

    allowed_signal_types: set[str] | None = None

    period: int = 20
    std_dev: float = 2.0
    price_col: str = "close"

    def __post_init__(self) -> None:
        self.allowed_signal_types = {"rise", "fall"}

        self.features = [
            _BollingerFeature(
                period=self.period,
                std_dev=self.std_dev,
                price_col=self.price_col,
            ),
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        df = features.sort([self.pair_col, self.ts_col])
        df = df.filter(pl.col("bb_upper").is_not_null() & pl.col("bb_lower").is_not_null())

        price = pl.col(self.price_col)
        price_prev = price.shift(1).over(self.pair_col)
        lower = pl.col("bb_lower")
        lower_prev = lower.shift(1).over(self.pair_col)
        upper = pl.col("bb_upper")
        upper_prev = upper.shift(1).over(self.pair_col)

        # Price crosses below lower band → buy (mean reversion up)
        touch_lower = (price <= lower) & (price_prev > lower_prev)
        # Price crosses above upper band → sell (mean reversion down)
        touch_upper = (price >= upper) & (price_prev < upper_prev)

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.when(touch_lower)
                .then(pl.lit("rise"))
                .when(touch_upper)
                .then(pl.lit("fall"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias("signal_type"),
                pl.when(touch_lower).then(1).when(touch_upper).then(-1).otherwise(0).alias("signal"),
            ]
        ).filter(pl.col("signal_type").is_not_null())

        return Signals(out)
