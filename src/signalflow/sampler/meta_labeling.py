"""MetaLabelingSampler - train only where a detector fired."""

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.decorators import sampler
from signalflow.enums import NONE, SIGNAL_COL, Provenance
from signalflow.sampler.base import Sampler, SampleSet


@sampler("meta_labeling")
class MetaLabelingSampler(Sampler):
    """Select rows where ``signal != NONE``."""

    def __init__(self, signals: Dataset):
        self.signals = signals

    def sample(self, data: Dataset) -> SampleSet:
        self._require_oos(self.signals)
        events = self.signals.frame.filter(pl.col(SIGNAL_COL) != NONE)
        weights = events.get_column("weight") if "weight" in events.columns else None
        return SampleSet(
            index=events.select(["pair", "ts"]),
            weights=weights,
            provenance=Provenance.OOS,
        )

    def to_config(self) -> dict:
        return {"sampler": "meta_labeling", "params": {"signals": "<runtime>"}}
