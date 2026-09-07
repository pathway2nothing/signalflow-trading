"""
SignalDetector - algorithmic fusion of forecasts into signals.

A detector is *not* a model: a transparent rule that combines forecast columns
(and raw features) into a discrete ``signal``. The base spans the whole range
from a trivial SMA cross to multi-forecast confluence - only ``detect()`` differs.

``run(data, forecasts=..., oos=True)`` is the standalone training-time helper:
it attaches each forecast slot's columns (via ``predict_oos`` when ``oos``) and
stamps the resulting Dataset's ``provenance`` so the MetaLabelingSampler's leak
guard (Invariant L) can verify it.
"""

from abc import abstractmethod
from typing import ClassVar

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.enums import SIGNAL_COL, Provenance
from signalflow.transform.base import Transform, ensure_sorted


class SignalDetector(Transform):
    """Transform whose single output column is ``signal``."""

    required_targets: ClassVar[dict] = {}
    """Slot -> acceptable registered target names; empty imposes no constraint."""

    learned: ClassVar[bool] = False
    """True for detectors that fit a model on the frame they detect on; their signals are in-sample."""

    @property
    def score_columns(self) -> list[str]:
        """Columns ``detect`` leaves on the frame that should travel with each signal.

        A forecast probability, a regime posterior, the indicator the rule fired
        on: whatever a strategy, a runner's persistence or a report wants to see
        next to ``signal`` without recomputing the detector. They are carried on
        emitted rows only (``enriched_signals``, ``Decision.signals``,
        ``Observation.signals``). Default: none.
        """
        return []

    @property
    def outputs(self) -> list[str]:
        return [SIGNAL_COL, *self.score_columns]

    def required_slots(self) -> "tuple[str, ...]":
        """Forecast slot names this detector reads; empty when it fuses none."""
        return ()

    @abstractmethod
    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append a ``signal`` column (RISE/FALL/NONE). Must be causal (use .over('pair'))."""

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.detect(ensure_sorted(df))

    def run(
        self,
        data: Dataset,
        forecasts: dict | None = None,
        oos: bool = False,
    ) -> Dataset:
        """Produce the signal event stream as a Dataset (carrying provenance)."""
        frame = data.frame
        if forecasts:
            for name, model in forecasts.items():
                pred = model.predict_oos(data) if oos else model.predict(data)
                out_col = getattr(model, "output", "p_rise")
                pred = pred.rename({out_col: f"{name}/{out_col}"})
                frame = frame.join(pred, on=["pair", "ts"], how="left")
        signals = self.compute(frame)
        provenance = Provenance.OOS if oos else Provenance.FULL
        return data.with_frame(signals, provenance=provenance)
