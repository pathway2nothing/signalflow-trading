"""Target specs - declare what a model predicts; labels are derived."""


from abc import ABC, abstractmethod

import polars as pl

from signalflow.data.dataset import Dataset

LABEL_COL = "label"


_TARGETS: dict[str, type] = {}


def register_target(name: str):
    def wrap(cls):
        cls._sf_name = name
        _TARGETS[name] = cls
        return cls

    return wrap


def make_target(name: str, **params):
    if name not in _TARGETS:
        raise KeyError(f"unknown target {name!r}; have {sorted(_TARGETS)}")
    return _TARGETS[name](**params)


class Target(ABC):
    """Derives labels from raw data for a model to learn."""

    @property
    def name(self) -> str:
        return getattr(self, "_sf_name", type(self).__name__)

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Forward bars consumed - used for purge/embargo in the walk-forward CV."""

    @abstractmethod
    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        """Return ``(pair, ts, label)``; if ``at`` (pair, ts) given, restrict to it."""

    def to_config(self) -> dict:
        import dataclasses

        params = {}
        if dataclasses.is_dataclass(self):
            params = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if not f.name.startswith("_")}
        return {"target": self.name, "params": params}

    @staticmethod
    def _restrict(labels: pl.DataFrame, at: pl.DataFrame | None) -> pl.DataFrame:
        if at is None:
            return labels
        return at.select(["pair", "ts"]).join(labels, on=["pair", "ts"], how="left")
