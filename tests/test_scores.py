"""Detector score columns travel with the signals they explain."""

import polars as pl
import pytest

import signalflow as sf
from signalflow.flow.loop import enriched_signals


class _ScoredDetector(sf.SignalDetector):
    """RISE when close is above its 10-bar mean; exposes the gap as a score."""

    @property
    def warmup(self) -> int:
        return 10

    @property
    def score_columns(self) -> list[str]:
        return ["gap"]

    def detect(self, df):
        gap = (pl.col("close") / pl.col("close").rolling_mean(10).over("pair") - 1.0).alias("gap")
        return df.with_columns(gap).with_columns(
            pl.when(pl.col("gap") > 0).then(pl.lit(sf.RISE)).otherwise(pl.lit(sf.NONE)).alias("signal")
        )


@pytest.fixture(scope="module")
def scored_ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2024-01-01", end="2024-02-01", interval="1h")


def test_scores_ride_along_on_emitted_rows(scored_ds):
    flow = sf.Flow(name="s", detectors=[_ScoredDetector()], strategy=sf.RulesStrategy())
    sig = enriched_signals(flow, scored_ds)
    assert "gap" in sig.columns and sig.height > 0
    assert sig.get_column("gap").null_count() == 0
    assert (sig.get_column("gap") > 0).all()
    assert sig.get_column("signal").unique().to_list() == [sf.RISE]
    assert _ScoredDetector().outputs == ["signal", "gap"]


def test_detectors_with_different_scores_align_with_nulls(scored_ds):
    flow = sf.Flow(
        name="two", detectors=[_ScoredDetector(), sf.SmaCrossDetector(fast=3, slow=20)], strategy=sf.RulesStrategy()
    )
    sig = enriched_signals(flow, scored_ds)
    scored = sig.filter(pl.col("detector") == "_ScoredDetector")
    cross = sig.filter(pl.col("detector") == "sma_cross")
    assert scored.height > 0 and cross.height > 0
    assert scored.get_column("gap").null_count() == 0
    assert cross.get_column("gap").null_count() == cross.height


def test_threshold_detector_carries_the_forecast_probability(ds, fitted_forecast):
    det = sf.ThresholdDetector(forecast="rise", p_min=0.5)
    assert det.score_columns == ["rise/p_rise"]
    flow = sf.Flow(name="t", forecasts={"rise": fitted_forecast}, detectors=[det], strategy=sf.RulesStrategy())
    sig = enriched_signals(flow, ds)
    if sig.height:
        assert "rise/p_rise" in sig.columns
        assert (sig.get_column("rise/p_rise") > 0.5).all()


def test_decision_and_observation_expose_scores(scored_ds):
    flow = sf.Flow(name="d", detectors=[_ScoredDetector()], strategy=sf.RulesStrategy())
    engine = sf.Engine(10_000, target="USDT")
    ts = scored_ds.frame.get_column("ts").max()
    decision = flow.decide(scored_ds, engine.snapshot(ts, {}))
    assert "gap" in decision.signals.columns
    obs = sf.Observation(ts, decision.signals, engine.snapshot(ts, {}), {})
    assert obs.score_columns == ["gap"]
    ctx = obs.to_prompt_context()
    if ctx["signals"]:
        assert "gap" in ctx["signals"][0]
