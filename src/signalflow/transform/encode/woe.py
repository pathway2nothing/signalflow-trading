"""WoE target-encoding - the default feature policy."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from signalflow.decorators import transform
from signalflow.enums import RESERVED_COLUMNS
from signalflow.errors import DegenerateTargetError
from signalflow.transform.base import Transform
from signalflow.transform.encode import stats


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
        if c not in RESERVED_COLUMNS and not c.endswith("__woe") and dt.is_numeric()
    ]


def _binarize(
    target: pl.Series,
    positive_threshold: float = 0.0,
    positive_classes: "tuple[float, ...] | None" = None,
) -> np.ndarray:
    """Coerce a target to a binary event mask; raise if it collapses to one class."""
    arr = target.to_numpy().astype(float)
    if positive_classes is not None:
        y = np.isin(arr, np.asarray(positive_classes, dtype=float)).astype(np.int64)
    else:
        y = (arr > positive_threshold).astype(np.int64)
    if np.unique(y).size < 2:
        raise DegenerateTargetError(
            f"binarization collapsed to a single class (positive_threshold={positive_threshold}, "
            f"positive_classes={positive_classes}); no Weight-of-Evidence can be estimated."
        )
    return y


@transform("woe")
@dataclass
class WoE(Transform):
    """Encode every feature column via Weight of Evidence against the target.

    ``refit`` is the rolling retrain cadence and ``window`` the trailing training span
    (duration strings such as ``"1d"``/``"365d"``); either set to ``None`` (or ``""`` for
    back-compat) disables rolling refit and fits once over all data.
    ``positive_threshold``/``positive_classes`` control how the target is binarized.
    """

    binning: Binning = field(default_factory=lambda: Binning("monotonic", 10))
    refit: str | None = "1d"
    window: str | None = "365d"
    smoothing: float = 0.5
    positive_threshold: float = 0.0
    positive_classes: tuple[float, ...] | None = None
    columns: list[str] | None = None

    requires_fit = True
    requires_target = True

    def __post_init__(self) -> None:
        if self.positive_classes is not None:
            self.positive_classes = tuple(float(c) for c in self.positive_classes)

    @property
    def outputs(self) -> list[str]:
        cols = getattr(self, "columns_", [])
        return [f"{c}__woe" for c in cols]

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "WoE":
        if target is None:
            raise ValueError("WoE.fit requires a target (requires_target=True)")
        y = _binarize(target, self.positive_threshold, self.positive_classes)
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
                "positive_threshold": self.positive_threshold,
                "positive_classes": list(self.positive_classes) if self.positive_classes is not None else None,
                "columns": self.columns,
            },
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "WoE":
        """Rebuild the recipe (hyperparameters); the fitted table is restored via load_state."""
        params = dict(cfg.get("params") or {})
        binning = params.get("binning")
        if isinstance(binning, dict):
            params["binning"] = Binning(**binning)
        return cls(**params)

    def state_dict(self) -> dict:
        """Portable fitted state: bin edges + WoE table + IV per column (JSON-able)."""
        self._require_fitted("woe_")
        return {
            "columns": list(self.columns_),
            "binning": self.binning.to_dict(),
            "smoothing": self.smoothing,
            "edges": {c: self.edges_[c].tolist() for c in self.columns_},
            "woe": {c: self.woe_[c].tolist() for c in self.columns_},
            "iv": {c: float(self.iv_[c]) for c in self.columns_},
        }

    def load_state(self, state: dict) -> "WoE":
        """Restore fitted edges/WoE/IV produced by :meth:`state_dict`."""
        self.columns_ = list(state["columns"])
        self.edges_ = {c: np.asarray(state["edges"][c], dtype=float) for c in self.columns_}
        self.woe_ = {c: np.asarray(state["woe"][c], dtype=float) for c in self.columns_}
        self.iv_ = {c: float(state["iv"][c]) for c in self.columns_}
        return self
