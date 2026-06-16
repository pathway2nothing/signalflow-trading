"""WoE target-encoding - the default feature policy."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from signalflow.decorators import transform
from signalflow.transform.base import Transform
from signalflow.transform.encode import stats

RESERVED = {"pair", "ts", "open", "high", "low", "close", "volume", "signal", "label", "_w", "weight"}


@dataclass
class Binning:
    """Bin-edge policy used by WoE/IV. ``monotonic`` enforces a monotone event rate."""

    method: str = "monotonic"
    max_bins: int = 10

    def fit_column(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.method == "quantile":
            return stats.quantile_edges(x, self.max_bins)
        return stats.monotonic_edges(x, y, self.max_bins)

    def to_dict(self) -> dict:
        return {"method": self.method, "max_bins": self.max_bins}


def _feature_columns(df: pl.DataFrame, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        return explicit
    return [
        c
        for c, dt in zip(df.columns, df.dtypes, strict=True)
        if c not in RESERVED and not c.endswith("__woe") and dt.is_numeric()
    ]


def _binarize(target: pl.Series) -> np.ndarray:
    arr = target.to_numpy()
    return (arr > 0).astype(np.int64)


@transform("woe")
@dataclass
class WoE(Transform):
    """Encode every feature column via Weight of Evidence against the target."""

    binning: Binning = field(default_factory=lambda: Binning("monotonic", 10))
    refit: str = "1d"
    window: str = "365d"
    smoothing: float = 0.5
    columns: list[str] | None = None

    requires_fit = True
    requires_target = True

    @property
    def outputs(self) -> list[str]:
        cols = getattr(self, "columns_", [])
        return [f"{c}__woe" for c in cols]

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "WoE":
        if target is None:
            raise ValueError("WoE.fit requires a target (requires_target=True)")
        y = _binarize(target)
        cols = _feature_columns(df, self.columns)
        self.columns_ = cols
        self.edges_ = {}
        self.woe_ = {}
        self.iv_ = {}
        for c in cols:
            x = df.get_column(c).to_numpy().astype(float)
            edges = self.binning.fit_column(x, y)
            bins = stats.assign_bins(x, edges)
            n_bins = edges.size + 1
            woe, iv = stats.compute_woe_table(bins, y, n_bins, self.smoothing)
            self.edges_[c] = edges
            self.woe_[c] = woe
            self.iv_[c] = iv
        return self

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        self._require_fitted("woe_")
        new_cols = []
        for c in self.columns_:
            x = df.get_column(c).to_numpy().astype(float)
            edges = self.edges_[c]
            woe = self.woe_[c]
            bins = stats.assign_bins(x, edges)
            n_bins = edges.size + 1
            idx = np.where(bins < 0, n_bins, bins)
            mapped = woe[idx]
            new_cols.append(pl.Series(f"{c}__woe", mapped))
        return df.with_columns(new_cols)

    def to_config(self) -> dict:
        return {
            "transform": "woe",
            "role": "encode",
            "params": {
                "binning": self.binning.to_dict(),
                "refit": self.refit,
                "window": self.window,
                "smoothing": self.smoothing,
                "columns": self.columns,
            },
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "WoE":
        """Rebuild the recipe (hyperparameters); the fitted table persists via the model pickle."""
        params = dict(cfg.get("params") or {})
        binning = params.get("binning")
        if isinstance(binning, dict):
            params["binning"] = Binning(**binning)
        return cls(**params)
