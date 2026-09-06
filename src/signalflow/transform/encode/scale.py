"""Scaler - fit-on-train column scaling (standard or robust), applied in place."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

from signalflow.decorators import transform
from signalflow.enums import RESERVED_COLUMNS
from signalflow.transform.base import Transform


def _candidate_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes, strict=True) if c not in RESERVED_COLUMNS and dt.is_numeric()]


@transform("scaler")
@dataclass
class Scaler(Transform):
    """Scale numeric feature columns with statistics fitted on the training rows.

    ``method="standard"`` subtracts the mean and divides by the standard deviation;
    ``"robust"`` subtracts the median and divides by the inter-quartile range. The
    columns keep their names, so the pipeline's outputs are unchanged. Fitted inside
    every walk-forward fold like any stateful transform.
    """

    method: str = "standard"
    columns: list[str] | None = None
    eps: float = 1e-12

    requires_fit = True

    def __post_init__(self) -> None:
        if self.method not in ("standard", "robust"):
            raise ValueError(f"Scaler.method must be 'standard' or 'robust', got {self.method!r}")

    @property
    def outputs(self) -> list[str]:
        return list(getattr(self, "columns_", self.columns or []))

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "Scaler":
        cols = self.columns if self.columns is not None else _candidate_columns(df)
        self.columns_ = list(cols)
        self.center_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        for c in self.columns_:
            x = df.get_column(c).to_numpy().astype(float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                center, scale = 0.0, 1.0
            elif self.method == "standard":
                center, scale = float(x.mean()), float(x.std())
            else:
                q1, med, q3 = np.percentile(x, [25.0, 50.0, 75.0])
                center, scale = float(med), float(q3 - q1)
            self.center_[c] = center
            self.scale_[c] = scale if scale > self.eps else 1.0
        self._is_fitted = True
        logger.trace(f"Scaler.fit({self.method}): {len(self.columns_)} columns on rows={df.height:,}")
        return self

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        self._require_fitted("scale_")
        exprs = [
            ((pl.col(c) - self.center_[c]) / self.scale_[c]).alias(c) for c in self.columns_ if c in df.columns
        ]
        return df.with_columns(exprs) if exprs else df

    def state_dict(self) -> dict:
        self._require_fitted("scale_")
        return {"method": self.method, "columns": list(self.columns_), "center": self.center_, "scale": self.scale_}

    def load_state(self, state: dict) -> "Scaler":
        self.columns_ = list(state["columns"])
        self.center_ = {c: float(v) for c, v in state["center"].items()}
        self.scale_ = {c: float(v) for c, v in state["scale"].items()}
        self._is_fitted = True
        return self
