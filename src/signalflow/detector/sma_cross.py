# src/signalflow/detector/sma_cross_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import RawDataType, Signals, SignalType, sf_component
from signalflow.detector import SignalDetector
from signalflow.feature import FeatureSet
from signalflow.feature.smoother import SmaExtractor


@dataclass
@sf_component(name="sma_cross")
class SmaCrossSignalDetector(SignalDetector):
    """
    SMA crossover signal detector.

    Signal rules (per pair, per timestamp):
      - RISE  : fast crosses above slow  (fast_t > slow_t) and (fast_{t-1} <= slow_{t-1})
      - FALL  : fast crosses below slow  (fast_t < slow_t) and (fast_{t-1} >= slow_{t-1})
      - NONE  : otherwise

    Output Signals columns:
      - pair, timestamp, signal_type, signal
      - signal: +1 for RISE, -1 for FALL, 0 for NONE
    """

    fast_period: int = 20
    slow_period: int = 50
    price_col: str = "close"

    fast_col: str | None = None
    slow_col: str | None = None

    def __post_init__(self) -> None:
        if self.fast_period <= 0 or self.slow_period <= 0:
            raise ValueError("fast_period and slow_period must be > 0")
        if self.fast_period >= self.slow_period:
            raise ValueError(f"fast_period must be < slow_period, got {self.fast_period} >= {self.slow_period}")

        self.fast_col = self.fast_col or f"sma_{self.fast_period}"
        self.slow_col = self.slow_col or f"sma_{self.slow_period}"

        self.feature_set = FeatureSet(
            extractors=[
                SmaExtractor(
                    offset_window=1,
                    sma_period=self.fast_period,
                    price_col=self.price_col,
                    out_col=self.fast_col,
                    use_resample=True,
                    raw_data_type=RawDataType.SPOT,
                ),
                SmaExtractor(
                    offset_window=1,
                    sma_period=self.slow_period,
                    price_col=self.price_col,
                    out_col=self.slow_col,
                    use_resample=True,
                    raw_data_type=RawDataType.SPOT,
                ),
            ]
        )

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        df = features.sort([self.pair_col, self.ts_col])

        if self.fast_col not in df.columns or self.slow_col not in df.columns:
            raise ValueError(
                f"Expected columns '{self.fast_col}' and '{self.slow_col}' in features. "
                f"Got: {df.columns}"
            )

        df = df.filter(pl.col(self.fast_col).is_not_null() & pl.col(self.slow_col).is_not_null())

        fast = pl.col(self.fast_col)
        slow = pl.col(self.slow_col)

        fast_prev = fast.shift(1).over(self.pair_col)
        slow_prev = slow.shift(1).over(self.pair_col)

        cross_up = (fast > slow) & (fast_prev <= slow_prev)
        cross_down = (fast < slow) & (fast_prev >= slow_prev)

        out = (
            df.select([self.pair_col, self.ts_col, self.fast_col, self.slow_col])
            .with_columns(
                pl.when(cross_up)
                .then(pl.lit(SignalType.RISE.value))
                .when(cross_down)
                .then(pl.lit(SignalType.FALL.value))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias("signal_type")
            )
            .with_columns(
                pl.when(pl.col("signal_type") == SignalType.RISE.value).then(pl.lit(1))
                .when(pl.col("signal_type") == SignalType.FALL.value).then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .alias("signal")
            )
        )

        return Signals(out.select([self.pair_col, self.ts_col, "signal_type", "signal"]))