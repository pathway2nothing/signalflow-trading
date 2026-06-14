"""Classic detectors - consume only feature/price columns, no forecasts."""


from dataclasses import dataclass

import polars as pl

from signalflow.decorators import detector
from signalflow.detector.base import SignalDetector
from signalflow.enums import NONE, RISE, SIGNAL_COL


@detector("sma_cross")
@dataclass
class SmaCrossDetector(SignalDetector):
    """RISE when the fast SMA crosses above the slow SMA."""

    fast: int = 20
    slow: int = 50

    @property
    def warmup(self) -> int:
        return self.slow + 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("close").rolling_mean(self.fast).over("pair").alias("_f"),
            pl.col("close").rolling_mean(self.slow).over("pair").alias("_s"),
        )
        cross_up = (pl.col("_f") > pl.col("_s")) & (
            pl.col("_f").shift(1).over("pair") <= pl.col("_s").shift(1).over("pair")
        )
        return df.with_columns(
            pl.when(cross_up).then(pl.lit(RISE)).otherwise(pl.lit(NONE)).alias(SIGNAL_COL)
        ).drop(["_f", "_s"])
