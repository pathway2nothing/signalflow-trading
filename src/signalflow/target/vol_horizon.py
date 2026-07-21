"""VolHorizon target - fixed horizon with a volatility-scaled dead zone."""

from dataclasses import dataclass

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target.base import LABEL_COL, Target, register_target, resolve_bars


@register_target("vol_horizon")
@dataclass
class VolHorizon(Target):
    """Label 1/0 when the forward return clears +/- k * horizon-scaled sigma; null inside the zone.

    ``sigma`` is the EWMA 1-bar return std scaled by ``sqrt(bars)`` so the dead
    zone is in units of the horizon's own volatility.
    """

    bars: int | str = 120
    k: float = 0.5
    vol_window: int = 100

    @property
    def horizon(self) -> int:
        return self.bars if isinstance(self.bars, int) else 1

    def _effective_horizon(self) -> "int | str":
        return self.bars

    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        bars = resolve_bars(self.bars, data)
        frame = (
            data.frame.sort(["pair", "ts"])
            .with_columns((pl.col("close") / pl.col("close").shift(1).over("pair") - 1.0).alias("_r1"))
            .with_columns(
                (pl.col("_r1").ewm_std(span=self.vol_window).over("pair") * float(bars) ** 0.5).alias("_sigma"),
                (pl.col("close").shift(-bars).over("pair") / pl.col("close") - 1.0).alias("_fwd"),
            )
        )
        label = (
            pl.when(pl.col("_fwd") > self.k * pl.col("_sigma"))
            .then(1)
            .when(pl.col("_fwd") < -self.k * pl.col("_sigma"))
            .then(0)
            .otherwise(None)
            .cast(pl.Int64)
        )
        labels = frame.with_columns(label.alias(LABEL_COL)).select(["pair", "ts", LABEL_COL])
        return self._restrict(labels, at)
