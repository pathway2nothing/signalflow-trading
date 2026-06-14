"""IVSelector - drop features below an Information Value threshold."""


from dataclasses import dataclass

import polars as pl

from signalflow.decorators import transform
from signalflow.transform.base import Transform
from signalflow.transform.encode import stats
from signalflow.transform.encode.woe import RESERVED, _binarize


def _candidate_columns(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c, dt in zip(df.columns, df.dtypes, strict=True)
        if c not in RESERVED and dt.is_numeric()
    ]


@transform("iv_selector")
@dataclass
class IVSelector(Transform):
    """Keep feature columns with IV ≥ ``min_iv``."""

    min_iv: float = 0.1
    max_bins: int = 10
    smoothing: float = 0.5

    requires_fit = True
    requires_target = True


    @property
    def outputs(self) -> list[str]:
        return getattr(self, "keep_", [])

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "IVSelector":
        if target is None:
            raise ValueError("IVSelector.fit requires a target")
        y = _binarize(target)
        self.iv_ = {}
        keep: list[str] = []
        for c in _candidate_columns(df):
            x = df.get_column(c).to_numpy().astype(float)
            edges = stats.quantile_edges(x, self.max_bins)
            bins = stats.assign_bins(x, edges)
            iv = stats.information_value(bins, y, edges.size + 1, self.smoothing)
            self.iv_[c] = iv
            if iv >= self.min_iv:
                keep.append(c)
        self.keep_ = keep
        return self

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        self._require_fitted("keep_")
        candidates = set(_candidate_columns(df))
        drop = [c for c in candidates if c not in self.keep_]
        return df.drop(drop)
