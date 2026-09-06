"""IVSelector - drop features below an Information Value threshold."""

from dataclasses import dataclass

import polars as pl
from loguru import logger

from signalflow._logging import names
from signalflow.decorators import transform
from signalflow.enums import RESERVED_COLUMNS
from signalflow.transform.base import Transform
from signalflow.transform.encode import stats
from signalflow.transform.encode.woe import _binarize


def _candidate_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes, strict=True) if c not in RESERVED_COLUMNS and dt.is_numeric()]


@transform("iv_selector")
@dataclass
class IVSelector(Transform):
    """Keep feature columns with IV ≥ ``min_iv``.

    ``positive_threshold``/``positive_classes`` control how the target is binarized.
    """

    min_iv: float = 0.1
    max_bins: int = 10
    smoothing: float = 0.5
    positive_threshold: float = 0.0
    positive_classes: tuple[float, ...] | None = None

    requires_fit = True
    requires_target = True
    narrows = True

    def __post_init__(self) -> None:
        if self.positive_classes is not None:
            self.positive_classes = tuple(float(c) for c in self.positive_classes)

    @property
    def outputs(self) -> list[str]:
        return getattr(self, "keep_", [])

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "IVSelector":
        if target is None:
            raise ValueError("IVSelector.fit requires a target")
        y = _binarize(target, self.positive_threshold, self.positive_classes)
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
        if not keep and self.iv_:
            logger.warning(
                f"IVSelector.fit: no column reached min_iv={self.min_iv}; keeping all {len(self.iv_)} candidates"
            )
            keep = list(self.iv_)
        self.keep_ = keep
        self._is_fitted = True
        dropped = [c for c in self.iv_ if c not in keep]
        logger.trace(
            f"IVSelector.fit: kept {len(keep)}/{len(self.iv_)} columns (min_iv={self.min_iv}); dropped {names(dropped)}"
        )
        return self

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        self._require_fitted("keep_")
        candidates = set(_candidate_columns(df))
        drop = [c for c in candidates if c not in self.keep_]
        return df.drop(drop)
