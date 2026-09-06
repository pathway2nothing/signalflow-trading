"""Golden regression values for the refactoring (SF_REFACTORING_PLAN.md, WP0).

These constants pin the numeric behaviour of the core on the deterministic
synthetic dataset. A refactoring work package must leave them unchanged unless
it deliberately changes behaviour - then the constants are updated in the same
change and the reason is recorded in CHANGELOG.md.
"""

import polars as pl
import pytest

import signalflow as sf

GOLDEN_ROWS = 2832
GOLDEN_CLOSE_SUM = 376989.908424

# ForecastModel(SMA20, SMA10, SMA50, WoE, IVSelector -> FixedHorizon(12), Rolling(1d, 365d)), 2-pair hourly data.
GOLDEN_OOS_ROWS = 2590
GOLDEN_OOS_P_SUM = 1137.256767
GOLDEN_PREDICT_P_SUM = 1278.285491  # WP2: null inside the warmup (98 rows with null SMA inputs)

# Flow(threshold p_min=0.6, RulesStrategy) backtest with capital 10_000.
# WP2: predict is null inside the warmup, so no trades happen there.
GOLDEN_MODEL_FLOW = {"n_fills": 9, "final_equity": 10016.43, "max_drawdown": 0.0055, "sharpe": 0.878}
GOLDEN_MODEL_FLOW_OOS = {"n_fills": 12, "final_equity": 9981.07, "oos_coverage": 0.9145}

# Flow(SmaCrossDetector(10, 30), RulesStrategy) backtest with capital 10_000.
GOLDEN_CROSS_FLOW = {"n_fills": 40, "final_equity": 9835.02, "max_drawdown": 0.0205, "sharpe": -4.446}

# TripleBarrier(tp=0.03, sl=0.015, max_bars=100) labels over the dataset.
GOLDEN_TRIPLE_BARRIER_POSITIVES = 254


@pytest.fixture(scope="module")
def golden_ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-03-01", interval="1h")


@pytest.fixture(scope="module")
def golden_model(golden_ds):
    model = sf.ForecastModel(
        backend="lightgbm",
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipeline(sf.SMA(20), sf.SMA(10), sf.SMA(50), sf.WoE(), sf.IVSelector()),
        output="p_rise",
        cv=sf.Rolling(step="1d", window="365d"),  # the pre-WP3 default: daily refit on a trailing year
    )
    return model.fit(golden_ds)


@pytest.fixture(scope="module")
def golden_flow(golden_model):
    return sf.Flow(
        name="golden",
        forecasts={"rise": golden_model},
        detectors=[sf.ThresholdDetector(forecast="rise", p_min=0.6)],
        strategy=sf.RulesStrategy(),
    )


def _subset(card: dict, keys) -> dict:
    return {k: card[k] for k in keys}


def test_golden_dataset(golden_ds):
    assert golden_ds.height == GOLDEN_ROWS
    assert round(float(golden_ds.frame.get_column("close").sum()), 6) == GOLDEN_CLOSE_SUM


def test_golden_oos_predictions(golden_model):
    oos = golden_model.oos_
    assert oos.height == GOLDEN_OOS_ROWS
    assert oos.get_column("p_rise").null_count() == 0
    assert round(float(oos.get_column("p_rise").sum()), 6) == GOLDEN_OOS_P_SUM


def test_golden_predict(golden_model, golden_ds):
    pred = golden_model.predict(golden_ds)
    assert round(float(pred.get_column("p_rise").sum()), 6) == GOLDEN_PREDICT_P_SUM


def test_golden_model_flow_backtest(golden_flow, golden_ds):
    card = golden_flow.backtest(golden_ds, capital=10_000).scorecard()
    assert _subset(card, GOLDEN_MODEL_FLOW) == GOLDEN_MODEL_FLOW
    assert card["promotable"] is False


def test_golden_model_flow_oos_backtest(golden_flow, golden_ds):
    card = golden_flow.backtest(golden_ds, capital=10_000, oos=True).scorecard()
    assert _subset(card, GOLDEN_MODEL_FLOW_OOS) == GOLDEN_MODEL_FLOW_OOS


def test_golden_backtest_equals_simulate(golden_flow, golden_ds):
    """Model flows agree with the live loop under the *default* warmup: predict is null inside it."""
    bt = golden_flow.backtest(golden_ds, capital=10_000)
    sim = golden_flow.simulate(golden_ds, capital=10_000)
    assert sim.final_equity == pytest.approx(bt.final_equity)
    assert len(sim.fills) == len(bt.fills)


def test_golden_cross_flow(golden_ds):
    flow = sf.Flow(name="cross", detectors=[sf.SmaCrossDetector(fast=10, slow=30)], strategy=sf.RulesStrategy())
    bt = flow.backtest(golden_ds, capital=10_000)
    assert _subset(bt.scorecard(), GOLDEN_CROSS_FLOW) == GOLDEN_CROSS_FLOW
    sim = flow.simulate(golden_ds, capital=10_000)
    assert sim.final_equity == pytest.approx(bt.final_equity)
    assert len(sim.fills) == len(bt.fills)


def test_golden_triple_barrier(golden_ds):
    labels = sf.TripleBarrier(tp=0.03, sl=0.015, max_bars=100).labels(golden_ds)
    assert labels.height == GOLDEN_ROWS
    assert int(labels.get_column("label").cast(pl.Int64).sum()) == GOLDEN_TRIPLE_BARRIER_POSITIVES
