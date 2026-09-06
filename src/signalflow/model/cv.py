"""Cross-validation schemes: how ``ForecastModel.fit`` produces its out-of-fold predictions.

The scheme decides *how many* embargoed walk-forward folds are fitted and how
each fold's training window is bounded. It is independent of the feature
encoder: an encoder such as ``WoE`` is simply refitted inside every fold.

* :class:`KFold` - ``n`` contiguous, equal blocks; each fold trains on
  everything before its block (expanding window).
* :class:`Rolling` - test blocks of ``step`` width; each fold trains on the
  trailing ``window`` (or on everything before, when ``window`` is ``None``).
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, runtime_checkable

from loguru import logger

from signalflow._time import parse_duration
from signalflow.model.oos import Fold, make_folds, rolling_folds

_FALLBACK_FOLDS = 3


@runtime_checkable
class CVScheme(Protocol):
    """Produces the walk-forward folds for one training span."""

    def folds(self, ts_unique: list, embargo: timedelta) -> list[Fold]: ...

    def to_config(self) -> dict: ...


@dataclass
class KFold:
    """``n`` contiguous blocks; fold ``k`` trains on blocks before ``k`` and tests on block ``k``."""

    n: int = 5

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"KFold needs n >= 2, got {self.n}")

    def folds(self, ts_unique: list, embargo: timedelta) -> list[Fold]:
        return make_folds(ts_unique, self.n)

    def to_config(self) -> dict:
        return {"scheme": "kfold", "n": self.n}


@dataclass
class Rolling:
    """Test blocks ``step`` wide, refitted each block on the trailing ``window``.

    ``window=None`` trains on everything before the block (expanding). Out-of-fold
    coverage loses roughly the first ``step`` of the span (nothing precedes it to
    train on), so pick a small step when promotion evidence needs ``>= 95 %``
    coverage on a short history. When the
    span is too short for a single rolling fold the scheme falls back to
    ``KFold(3)`` with a warning, mirroring what the plain block split does on
    tiny datasets.
    """

    step: str = "7d"
    window: str | None = "365d"

    def __post_init__(self) -> None:
        parse_duration(self.step)
        if self.window is not None:
            parse_duration(self.window)

    def folds(self, ts_unique: list, embargo: timedelta) -> list[Fold]:
        step = parse_duration(self.step)
        window = parse_duration(self.window) if self.window else None
        folds = rolling_folds(ts_unique, step, window, embargo)
        if folds:
            return folds
        logger.warning(
            f"Rolling(step={self.step!r}): span too short for a rolling fold; falling back to KFold({_FALLBACK_FOLDS})"
        )
        return make_folds(ts_unique, _FALLBACK_FOLDS)

    def to_config(self) -> dict:
        return {"scheme": "rolling", "step": self.step, "window": self.window}


def build_cv(cfg: "dict | CVScheme | None") -> CVScheme:
    """Rebuild a scheme from ``to_config`` output (``{"scheme": "kfold", "n": 5}`` /
    ``{"scheme": "rolling", "step": "30d", "window": "365d"}``); ``None`` gives the default."""
    if cfg is None:
        return Rolling()
    if not isinstance(cfg, dict):
        return cfg
    scheme = str(cfg.get("scheme", "rolling")).lower()
    if scheme == "kfold":
        return KFold(n=int(cfg.get("n", 5)))
    if scheme == "rolling":
        return Rolling(step=str(cfg.get("step", "7d")), window=cfg.get("window", "365d"))
    raise ValueError(f"unknown cv scheme {scheme!r}; use 'kfold' or 'rolling'")


__all__ = ["CVScheme", "KFold", "Rolling", "build_cv"]
