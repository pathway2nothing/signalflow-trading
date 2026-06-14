"""FeaturePipe - an ordered, nestable composition of Transforms."""


import polars as pl

from signalflow.enums import SIGNAL_COL
from signalflow.errors import PipeError
from signalflow.transform.base import Transform


class FeaturePipe(Transform):
    """Run child transforms in order; outputs are the union of their outputs."""

    def __init__(self, *transforms: Transform):
        for t in transforms:
            if SIGNAL_COL in t.outputs:
                raise PipeError(
                    f"{t.name!r} outputs {SIGNAL_COL!r}; detectors go in detectors=, "
                    "not in a FeaturePipe (signal-as-feature is a separate explicit step)"
                )
        self.transforms: tuple[Transform, ...] = transforms

    @property
    def warmup(self) -> int:
        return max((t.warmup for t in self.transforms), default=0)

    @property
    def outputs(self) -> list[str]:
        out: list[str] = []
        for t in self.transforms:
            out.extend(t.outputs)
        return out

    @property
    def requires_fit(self) -> bool:
        return any(t.requires_fit for t in self.transforms)

    @property
    def requires_target(self) -> bool:
        return any(t.requires_target for t in self.transforms)

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "FeaturePipe":
        """Fit stateful children in order, each on the frame produced so far."""
        cur = df
        for t in self.transforms:
            if t.requires_fit:
                t.fit(cur, target if t.requires_target else None)
            cur = t.compute(cur)
        return self

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        cur = df
        for t in self.transforms:
            cur = t.compute(cur)
        return cur

    def to_config(self) -> dict:
        return {
            "transform": "feature_pipe",
            "role": "pipe",
            "params": {"transforms": [t.to_config() for t in self.transforms]},
        }
