"""Validator combinators."""


from functools import reduce

import polars as pl

from signalflow.data.dataset import Dataset


class _Combinator:
    output = "p_success"

    def __init__(self, children: list):
        self.children = children

    @property
    def is_fitted(self) -> bool:
        return all(getattr(c, "is_fitted", False) for c in self.children)

    def fit(self, data: Dataset, sampler=None):
        for c in self.children:
            if not getattr(c, "is_fitted", False):
                c.fit(data, sampler)
        return self

    def _child_frames(self, data: Dataset, oos: bool) -> list[pl.DataFrame]:
        frames = []
        for i, c in enumerate(self.children):
            pred = c.predict_oos(data) if oos else c.predict(data)
            col = c.output if hasattr(c, "output") else "p_rise"
            frames.append(pred.rename({col: f"_c{i}"}).select(["pair", "ts", f"_c{i}"]))
        return frames

    def _join(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        return reduce(lambda a, b: a.join(b, on=["pair", "ts"], how="outer_coalesce"), frames)

    def _aggregate(self, joined: pl.DataFrame, value_cols: list[str]) -> pl.Series:
        raise NotImplementedError

    def _predict(self, data: Dataset, oos: bool) -> pl.DataFrame:
        frames = self._child_frames(data, oos)
        joined = self._join(frames)
        value_cols = [c for c in joined.columns if c.startswith("_c")]
        agg = self._aggregate(joined, value_cols)
        return joined.select(["pair", "ts"]).with_columns(agg.alias(self.output))

    def predict(self, data: Dataset) -> pl.DataFrame:
        return self._predict(data, oos=False)

    def predict_oos(self, data: Dataset) -> pl.DataFrame:
        return self._predict(data, oos=True)


class MeanValidator(_Combinator):
    """Average child probabilities."""

    def _aggregate(self, joined, value_cols):
        return pl.mean_horizontal([pl.col(c) for c in value_cols])


class MaxValidator(_Combinator):
    """Take the maximum child probability."""

    def _aggregate(self, joined, value_cols):
        return pl.max_horizontal([pl.col(c) for c in value_cols])


class VoteValidator(_Combinator):
    """Fraction of children whose probability exceeds ``threshold``."""

    def __init__(self, children: list, threshold: float = 0.5):
        super().__init__(children)
        self.threshold = threshold

    def _aggregate(self, joined, value_cols):
        votes = [(pl.col(c) > self.threshold).cast(pl.Float64) for c in value_cols]
        return pl.sum_horizontal(votes) / len(value_cols)
