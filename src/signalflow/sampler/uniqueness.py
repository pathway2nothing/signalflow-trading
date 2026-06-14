"""UniquenessSampler - average-uniqueness weights over a base sampler."""


import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.decorators import sampler
from signalflow.sampler.base import Sampler, SampleSet
from signalflow.sampler.uniform import UniformSampler


@sampler("uniqueness")
class UniquenessSampler(Sampler):
    """Add average-uniqueness weights to ``base``'s selection for a horizon target."""

    def __init__(self, target, base: Sampler | None = None):
        self.target = target
        self.base = base or UniformSampler()

    def sample(self, data: Dataset) -> SampleSet:
        ss = self.base.sample(data)
        horizon = max(1, int(getattr(self.target, "horizon", 1)))

        idx = ss.index.sort(["pair", "ts"]).with_row_index("_i")
        weights = []
        for _pair, sub in idx.group_by("pair", maintain_order=True):
            ts = sub.get_column("ts").to_list()
            n = len(ts)
            for i in range(n):

                lo = max(0, i - horizon)
                hi = min(n, i + horizon + 1)
                concurrency = hi - lo
                weights.append(1.0 / concurrency)
        return SampleSet(index=ss.index, weights=pl.Series("weight", weights), provenance=ss.provenance)

    def to_config(self) -> dict:
        return {
            "sampler": "uniqueness",
            "params": {"target": self.target.to_config(), "base": self.base.to_config()},
        }
