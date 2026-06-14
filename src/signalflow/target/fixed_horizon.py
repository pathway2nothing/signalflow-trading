"""FixedHorizon target - does price rise over the next N bars?"""


from dataclasses import dataclass

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target.base import LABEL_COL, Target, register_target


@register_target("fixed_horizon")
@dataclass
class FixedHorizon(Target):
    """Label 1 if close rises by more than ``threshold`` after ``bars`` bars."""

    bars: int = 120
    threshold: float = 0.0

    @property
    def horizon(self) -> int:
        return self.bars

    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        fut_ret = (pl.col("close").shift(-self.bars) / pl.col("close") - 1.0).over("pair")
        labels = (
            data.frame.sort(["pair", "ts"])
            .with_columns((fut_ret > self.threshold).cast(pl.Int64).alias(LABEL_COL))
            .select(["pair", "ts", LABEL_COL])
        )
        return self._restrict(labels, at)
