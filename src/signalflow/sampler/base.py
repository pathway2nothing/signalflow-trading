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
    def sample(self, data: Dataset) -> SampleSet:
        """Select training rows from ``data``, returning their index, optional weights,
        and a provenance stamp."""
        ...

    @staticmethod
    def _require_oos(data: "Dataset", columns: "list[str] | None" = None) -> None:
        """Verify forecast provenance is OOS before training on it.

        Prefers per-column provenance when tracked; otherwise falls back to the
        frame-level tag for backward compatibility.
        """
        col_prov = getattr(data, "col_provenance", None) or {}
        tracked = {c: col_prov[c] for c in (columns if columns is not None else col_prov) if c in col_prov}
        if tracked:
            offenders = [c for c, p in tracked.items() if Provenance(p) != Provenance.OOS]
            if offenders:
                raise LeakageError(
                    f"training set built on non-OOS forecast column(s) {offenders}; "
                    f"attach predictions with Dataset.with_oos_forecasts(model)"
                )
            return
        if Provenance(data.provenance) != Provenance.OOS:
            raise LeakageError(
                "training set built on in-sample forecasts; use detector.run(data, oos=True) "
                "or Dataset.with_oos_forecasts(model)"
            )

    def to_config(self) -> dict:
        import dataclasses

        params = {}
        if dataclasses.is_dataclass(self):
            params = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if not f.name.startswith("_")}
        return {"sampler": self.name, "params": params}
