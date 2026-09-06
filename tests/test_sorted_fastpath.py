"""Feature pipes sort once and fuse expression features; forecasts attach without a join when aligned."""

from dataclasses import dataclass

import polars as pl

import signalflow as sf
from signalflow.transform.base import Feature, Transform, ensure_sorted, is_sorted_pair_ts


@dataclass
class _Lag(Feature):
    """Depends on an earlier pipe output (sma_5) - exercises sequential semantics inside the lazy chain."""

    n: int = 1

    @property
    def outputs(self):
        return [f"sma5_lag{self.n}"]

    def exprs(self):
        return [pl.col("sma_5").shift(self.n).alias(f"sma5_lag{self.n}")]


@dataclass
class _EagerDouble(Transform):
    """Custom compute (not exprs) - must see the fused outputs computed before it."""

    @property
    def outputs(self):
        return ["sma5_x2"]

    def compute(self, df):
        return df.with_columns((pl.col("sma_5") * 2).alias("sma5_x2"))


def _ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-01-10", interval="1h")


def test_ensure_sorted_is_identity_on_sorted_frames():
    frame = _ds().frame
    assert is_sorted_pair_ts(frame)
    assert ensure_sorted(frame) is frame
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=1)
    assert not is_sorted_pair_ts(shuffled)
    assert ensure_sorted(shuffled).equals(frame)


def test_feature_on_shuffled_input_matches_sorted():
    frame = _ds().frame
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=2)
    assert sf.SMA(5).compute(shuffled).equals(sf.SMA(5).compute(frame))


def test_pipe_fused_output_matches_sequential_eager():
    frame = _ds().frame
    pipe = sf.FeaturePipeline(sf.SMA(5), _Lag(1), _EagerDouble(), sf.SMA(10), _Lag(2))
    fused = pipe.compute(frame)

    cur = frame
    for t in pipe.transforms:
        cur = t.compute(cur)
    assert fused.equals(cur)
    assert fused.columns[-5:] == ["sma_5", "sma5_lag1", "sma5_x2", "sma_10", "sma5_lag2"]
    # per-pair causality survives fusion: the first lagged row of each pair is null
    firsts = fused.group_by("pair", maintain_order=True).first()
    assert firsts.get_column("sma5_lag1").null_count() == 2


def test_with_forecasts_aligned_fast_path_matches_join():
    ds = _ds()
    pred = (
        ds.frame.select(["pair", "ts"])
        .with_row_index("_i")
        .with_columns((pl.col("_i") / 1000).alias("p_rise"))
        .drop("_i")
    )
    fast = ds.with_forecasts(pred)
    joined = ds.frame.join(pred, on=["pair", "ts"], how="left")
    assert fast.frame.equals(joined)
    assert fast.col_provenance == {"p_rise": "full"}

    # misaligned (shuffled, subset) predictions still align through the join
    partial = pred.sample(fraction=0.5, shuffle=True, seed=3)
    slow = ds.with_forecasts(partial)
    assert slow.frame.equals(ds.frame.join(partial, on=["pair", "ts"], how="left"))
    assert slow.frame.get_column("p_rise").null_count() == ds.height - partial.height
