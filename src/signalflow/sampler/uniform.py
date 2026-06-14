"""UniformSampler - every bar (default for forecast models)."""


from signalflow.data.dataset import Dataset
from signalflow.decorators import sampler
from signalflow.enums import Provenance
from signalflow.sampler.base import Sampler, SampleSet


@sampler("uniform")
class UniformSampler(Sampler):
    """Train on every row."""

    def sample(self, data: Dataset) -> SampleSet:
        return SampleSet(index=data.index(), provenance=Provenance.FULL)
