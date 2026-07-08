"""Forecast-fusion detectors - combine forecast slot columns into signals."""

from dataclasses import dataclass

import polars as pl

from signalflow.decorators import detector
from signalflow.detector.base import SignalDetector
from signalflow.enums import FALL, NONE, RISE, SIGNAL_COL


@detector("threshold")
@dataclass
class ThresholdDetector(SignalDetector):
    """RISE when a forecast's probability exceeds ``p_min``."""

    forecast: str = ""
    p_min: float = 0.6
    output: str = "p_rise"

    def required_slots(self) -> "tuple[str, ...]":
        return (self.forecast,) if self.forecast else ()

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        rise = pl.col(f"{self.forecast}/{self.output}") > self.p_min
        return df.with_columns(pl.when(rise).then(pl.lit(RISE)).otherwise(pl.lit(NONE)).alias(SIGNAL_COL))


@detector("revert")
@dataclass
class RevertDetector(SignalDetector):
    """RISE when the revert forecast is confident and (optionally) the regime is calm."""

    revert: str = ""
    regime: str | None = None
    p_min: float = 0.65
    regime_max: float = 0.3

    def required_slots(self) -> "tuple[str, ...]":
        slots = [s for s in (self.revert, self.regime) if s]
        return tuple(slots)

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        cond = pl.col(f"{self.revert}/p_rise") > self.p_min
        if self.regime:
            cond = cond & (pl.col(f"{self.regime}/p_turbulent") < self.regime_max)
        return df.with_columns(pl.when(cond).then(pl.lit(RISE)).otherwise(pl.lit(NONE)).alias(SIGNAL_COL))


@detector("market_drop")
@dataclass
class MarketDropDetector(SignalDetector):
    """FALL (exit/short seam) when a drop forecast is confident."""

    drop: str = ""
    p_min: float = 0.6
    output: str = "p_drop"

    def required_slots(self) -> "tuple[str, ...]":
        return (self.drop,) if self.drop else ()

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        fall = pl.col(f"{self.drop}/{self.output}") > self.p_min
        return df.with_columns(pl.when(fall).then(pl.lit(FALL)).otherwise(pl.lit(NONE)).alias(SIGNAL_COL))
