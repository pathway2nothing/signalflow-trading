"""CUSUMSampler - Lopez de Prado symmetric CUSUM filter events."""


from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.decorators import sampler
from signalflow.enums import Provenance
from signalflow.sampler.base import Sampler, SampleSet


@sampler("cusum")
@dataclass
class CUSUMSampler(Sampler):
    """Sample at structural shifts where cumulative log-return exceeds a threshold."""

    column: str = "close"
    threshold: float = 0.05

    def sample(self, data: Dataset) -> SampleSet:
        rows = []
        for pair, sub in data.frame.sort(["pair", "ts"]).group_by("pair", maintain_order=True):
            pair_name = pair[0] if isinstance(pair, tuple) else pair
            x = np.log(sub.get_column(self.column).to_numpy())
            diff = np.diff(x, prepend=x[:1])
            s_pos = s_neg = 0.0
            ts = sub.get_column("ts").to_list()
            for i in range(len(diff)):
                s_pos = max(0.0, s_pos + diff[i])
                s_neg = min(0.0, s_neg + diff[i])
                if s_pos > self.threshold:
                    s_pos = 0.0
                    rows.append((pair_name, ts[i]))
                elif s_neg < -self.threshold:
                    s_neg = 0.0
                    rows.append((pair_name, ts[i]))
        index = pl.DataFrame(rows, schema=["pair", "ts"], orient="row") if rows else data.index().head(0)
        return SampleSet(index=index, provenance=Provenance.FULL)
