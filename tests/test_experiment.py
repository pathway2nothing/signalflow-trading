"""Experiment lifecycle tests."""


import warnings

import polars as pl
import pytest

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")

import signalflow as sf
from signalflow.experiment import (
    ArtifactCache,
    Experiment,
    Scorecard,
    bootstrap_ci,
    monte_carlo_bounds,
)
from signalflow.experiment.scorecard import SCORECARD_KEYS


@pytest.fixture(scope="module")
def fitted_flow():
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")
    fc = sf.ForecastModel(
        backend="lightgbm",
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipe(sf.SMA(20), sf.SMA(10)),
        n_folds=3,
    )
    fc.fit(ds)
    flow = sf.Flow(
        name="e_flow",
        forecasts={"m": fc},
        detectors=[sf.ThresholdDetector(forecast="m", p_min=0.5)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.1), exit=sf.Exit(tp=0.03, sl=0.015)),
        risk=sf.Risk(max_drawdown=0.5, max_positions=2),
    )
    return flow, ds


def test_experiment_run_returns_run(fitted_flow):
    flow, ds = fitted_flow
    exp = Experiment("e", baseline=None)
    run = exp.run(flow, ds, capital=50_000)
    assert hasattr(run, "equity_curve")
    assert run.equity_curve.height > 0


def test_scorecard_shape(fitted_flow):
    flow, ds = fitted_flow
    exp = Experiment("e", baseline=None)
    exp.run(flow, ds, capital=50_000)
    card = exp.scorecard()
    for key in SCORECARD_KEYS:
        assert key in card, f"missing key {key}"
    assert "low" in card["bootstrap_ci"] and "high" in card["bootstrap_ci"]
    assert "p5" in card["monte_carlo"] and "p95" in card["monte_carlo"]
    assert card["delta"] is None


def test_scorecard_with_baseline_flow(fitted_flow):
    flow, ds = fitted_flow
    exp = Experiment("e", baseline=flow)
    exp.run(flow, ds, capital=50_000)
    card = exp.scorecard()
    assert card["delta"] is not None

    assert card["delta"]["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_scorecard_from_run_directly(fitted_flow):
    flow, ds = fitted_flow
    run = flow.backtest(ds, capital=50_000)
    card = Scorecard.from_run(run)
    assert set(SCORECARD_KEYS).issubset(card.keys())


def test_lifecycle_timestamps_explicit(fitted_flow):
    flow, ds = fitted_flow
    exp = Experiment("e")
    exp.run(flow, ds, capital=50_000, ts="2026-06-14T00:00:00")
    assert exp.log["created"] == "2026-06-14T00:00:00"
    assert exp.log["first_result"] == "2026-06-14T00:00:00"
    assert exp.log["stages"]


def test_compute_cached_calls_fn_once(tmp_path):
    cache = ArtifactCache(str(tmp_path))
    counter = {"n": 0}

    def make():
        counter["n"] += 1
        return pl.DataFrame({"a": [1, 2, 3]})

    key = cache.key({"x": 1}, kind="inference")
    df1 = cache.compute_cached(key, make)
    df2 = cache.compute_cached(key, make)
    assert counter["n"] == 1
    assert df1.equals(df2)


def test_key_changes_with_parts(tmp_path):
    cache = ArtifactCache(str(tmp_path))
    k1 = cache.key({"x": 1}, kind="inference")
    k2 = cache.key({"x": 2}, kind="inference")
    assert k1 != k2


def test_key_distinguishes_kind(tmp_path):
    cache = ArtifactCache(str(tmp_path))
    k_inf = cache.key({"x": 1}, kind="inference")
    k_oos = cache.key({"x": 1}, kind="oos_for_training")
    assert k_inf != k_oos


def test_key_changes_with_code_fingerprint(tmp_path):
    cache = ArtifactCache(str(tmp_path))

    def producer_a():
        return pl.DataFrame({"a": [1]})

    def producer_b():

        return pl.DataFrame({"a": [1, 2, 3, 4]})

    k_a = cache.key({"x": 1}, producer=producer_a)
    k_b = cache.key({"x": 1}, producer=producer_b)
    assert k_a != k_b


def test_invalid_kind_raises(tmp_path):
    cache = ArtifactCache(str(tmp_path))
    with pytest.raises(sf.ArtifactError):
        cache.key({"x": 1}, kind="bogus")


def test_bootstrap_ci_deterministic():
    rng_input = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02]
    a = bootstrap_ci(rng_input, n=500, seed=42)
    b = bootstrap_ci(rng_input, n=500, seed=42)
    assert a == b
    assert a[0] <= a[1]


def test_bootstrap_ci_seed_changes_result():
    rng_input = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02]
    a = bootstrap_ci(rng_input, n=500, seed=1)
    b = bootstrap_ci(rng_input, n=500, seed=2)
    assert a != b


def test_monte_carlo_deterministic():
    rng_input = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02]
    a = monte_carlo_bounds(rng_input, n=500, horizon=20, seed=7)
    b = monte_carlo_bounds(rng_input, n=500, horizon=20, seed=7)
    assert a == b
    assert a["p5"] <= a["p50"] <= a["p95"]


def test_stats_handle_empty():
    assert bootstrap_ci([], n=10) == (0.0, 0.0)
    mc = monte_carlo_bounds([], n=10)
    assert mc["p50"] == 1.0
