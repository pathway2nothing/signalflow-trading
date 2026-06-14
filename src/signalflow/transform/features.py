"""Core's single example feature. The full indicator library lives in signalflow-ta."""


from dataclasses import dataclass

import polars as pl

from signalflow.decorators import feature
from signalflow.transform.base import Feature

_CLOSE = pl.col("close")


@feature("sma")
@dataclass
class SMA(Feature):
    """Simple moving average of close."""

    length: int = 20

    @property
    def warmup(self) -> int:
        return self.length

    @property
    def outputs(self) -> list[str]:
        return [f"sma_{self.length}"]

    def exprs(self) -> list[pl.Expr]:
        return [_CLOSE.rolling_mean(self.length).alias(f"sma_{self.length}")]
