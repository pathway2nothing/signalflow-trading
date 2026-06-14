"""Sampler contract - which points to train on, and with what weights."""


from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.enums import Provenance
from signalflow.errors import LeakageError


@dataclass
class SampleSet:
    """Result of a sampler: training points + weights + provenance."""

    index: pl.DataFrame
    weights: pl.Series | None = None
    provenance: Provenance = Provenance.FULL


class Sampler(ABC):
    """Selects training points from a Dataset."""

    @property
    def name(self) -> str:
        return getattr(self, "_sf_name", type(self).__name__)

    @abstractmethod
    def sample(self, data: Dataset) -> SampleSet: ...

    @staticmethod
    def _require_oos(provenance: Provenance) -> None:
        if Provenance(provenance) != Provenance.OOS:
            raise LeakageError(
                "training set built on in-sample forecasts; use detector.run(data, oos=True)"
            )

    def to_config(self) -> dict:
        import dataclasses

        params = {}
        if dataclasses.is_dataclass(self):
            params = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if not f.name.startswith("_")}
        return {"sampler": self.name, "params": params}
