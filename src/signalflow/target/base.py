"""Target specs - declare what a model predicts; labels are derived."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.enums import ComponentType
from signalflow.registry import registry

LABEL_COL = "label"


def register_target(name: str) -> Callable[[type], type]:
    """Register a Target subclass in the central registry under ``name``."""

    def wrap(cls: type) -> type:
        cls._sf_name = name
        cls._sf_role = "target"
        cls._sf_type = ComponentType.TARGET
        bucket = registry._items.get(ComponentType.TARGET, {})
        existing = bucket.get(name.strip().lower())
        if existing is None or existing.cls is not cls:
            registry.register(ComponentType.TARGET, name, cls, role="target")
        return cls

    return wrap


def make_target(name: str, **params: object) -> "Target":
    """Construct a registered target by name."""
    return registry.create(ComponentType.TARGET, name, **params)


def _median_bar_seconds(data: Dataset) -> float:
    """Median spacing in seconds between consecutive bars of a dataset."""
    frame = data.frame
    if "ts" in frame.columns and frame.height >= 2:
        ts = frame.get_column("ts").unique().sort()
        if ts.len() >= 2:
            secs = ts.diff().drop_nulls().dt.total_seconds()
            positive = secs.filter(secs > 0)
            if positive.len() > 0:
                return float(positive.median())
    interval = data.source_params.get("interval")
    if interval:
        from signalflow.model.oos import parse_duration

        return parse_duration(interval).total_seconds()
    return 0.0


def resolve_bars(value: int | str, data: Dataset) -> int:
    """Resolve a bar count (int, passthrough) or a duration string against a dataset's interval."""
    if isinstance(value, bool):
        raise TypeError("horizon must be an int bar count or a duration string, not bool")
    if isinstance(value, int):
        return value
    from signalflow.model.oos import parse_duration

    seconds = parse_duration(value).total_seconds()
    bar_seconds = _median_bar_seconds(data)
    if bar_seconds <= 0:
        raise ValueError(f"cannot resolve duration {value!r}: dataset bar interval is unknown")
    return math.ceil(seconds / bar_seconds)


class Target(ABC):
    """Derives labels from raw data for a model to learn."""

    @property
    def name(self) -> str:
        return getattr(self, "_sf_name", type(self).__name__)

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Forward bars consumed - used for purge/embargo in the walk-forward CV."""

    def _effective_horizon(self) -> int | str:
        """Raw forward-look value (int bars or duration string) before bar resolution."""
        return self.horizon

    def horizon_bars(self, data: Dataset) -> int:
        """Forward bars for purge/embargo, resolving a duration-string horizon against the data."""
        return resolve_bars(self._effective_horizon(), data)

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
