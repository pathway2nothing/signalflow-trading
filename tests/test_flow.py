"""End-to-end Flow: backtest, leakage invariant, yaml round-trip, quicktest."""

from typing import ClassVar

import pytest

import signalflow as sf


@pytest.fixture(scope="module")
def flow(ds, fitted_forecast):
    return sf.Flow(
        name="e2e",
        forecasts={"revert": fitted_forecast},
        detectors=[sf.ThresholdDetector(forecast="revert", p_min=0.5)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.1), exit=sf.Exit(tp=0.03, sl=0.015)),
        risk=sf.Risk(max_drawdown=0.5, max_positions=2, max_notional_per_pair=0.3),
    )


def test_backtest_runs(flow, ds):
    run = flow.backtest(ds, capital=50_000)
    sc = run.scorecard()
    assert sc["promotable"] is False
    assert sc["oos"] is False
    assert run.equity_curve.height > 0
    assert sc["initial_equity"] == 50_000.0


def test_backtest_oos_differs_and_promotable(flow, ds):
    insample = flow.backtest(ds, capital=50_000)
    oos = flow.backtest(ds, capital=50_000, oos=True)
    assert insample.promotable is False and insample.oos is False
    assert oos.promotable is True and oos.oos is True
    assert oos.final_equity != insample.final_equity


def test_detector_only_flow_promotable(ds):
    f = sf.Flow(
        name="det_only",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.1), exit=sf.Exit(tp=0.03, sl=0.015)),
    )
    run = f.backtest(ds, capital=50_000)
    assert run.promotable is True


def test_quicktest_not_promotable(flow, ds):
    run = flow.quicktest(ds, capital=50_000)
    assert run.promotable is False


def test_quicktest_forwards_fee(ds):
    f = sf.Flow(
        name="qt",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.5)),
    )
    cheap = f.quicktest(ds, capital=50_000, fee=0.0).final_equity
    pricey = f.quicktest(ds, capital=50_000, fee=0.05).final_equity
    assert cheap != pricey


def test_untrained_flow_raises(ds):
    unfit = sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)))
    with pytest.raises(sf.UntrainedModelError):
        sf.Flow(name="bad", forecasts={"x": unfit})


def test_leakage_invariant(ds, fitted_forecast):
    """Invariant L: training a validator on in-sample (full) signals must raise."""
    det = sf.ThresholdDetector(forecast="revert", p_min=0.5)
    bad = det.run(ds, forecasts={"revert": fitted_forecast}, oos=False)
    meta = sf.ForecastModel(target=sf.TripleBarrier(), features=sf.FeaturePipe(sf.SMA(20)))
    with pytest.raises(sf.LeakageError):
        meta.fit(ds, sampler=sf.MetaLabelingSampler(signals=bad))


def test_oos_signals_train_validator(ds, fitted_forecast):
    det = sf.ThresholdDetector(forecast="revert", p_min=0.5)
    good = det.run(ds, forecasts={"revert": fitted_forecast}, oos=True)
    assert good.provenance == sf.Provenance.OOS
    meta = sf.ForecastModel(
        target=sf.TripleBarrier(tp=0.02, sl=0.01, max_bars=24),
        features=sf.FeaturePipe(sf.SMA(20), sf.SMA(50)),
        output="p_success",
        n_folds=3,
    )
    meta.fit(ds, sampler=sf.MetaLabelingSampler(signals=good))
    assert meta.is_fitted


def _equity_run(interval_seconds: int) -> "sf.Run":
    import datetime

    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=200)
    equity = 10_000.0 * np.cumprod(1.0 + returns)
    ts = [datetime.datetime(2023, 1, 1) + datetime.timedelta(seconds=interval_seconds * i) for i in range(len(equity))]
    curve = pl.DataFrame({"ts": ts, "equity": equity})
    return sf.Run(name="s", mode="backtest", equity_curve=curve)


def test_sharpe_scales_with_bar_interval():
    import math

    run_1m = _equity_run(60)
    run_1h = _equity_run(3600)
    ratio = run_1m.sharpe() / run_1h.sharpe()
    assert math.isclose(ratio, math.sqrt(60.0), rel_tol=1e-9)


def test_sharpe_fallback_when_too_few_points():
    import datetime

    import polars as pl

    curve = pl.DataFrame(
        {"ts": [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 1, 1)], "equity": [10_000.0, 10_100.0]}
    )
    run = sf.Run(name="s", mode="backtest", equity_curve=curve)
    assert run.periods_per_year() == 8760.0


def test_detector_missing_slot_raises(fitted_forecast):
    det = sf.ThresholdDetector(forecast="wrong", p_min=0.5)
    with pytest.raises(sf.FlowConfigError) as exc:
        sf.Flow(name="bad_slot", forecasts={"revert": fitted_forecast}, detectors=[det])
    assert "wrong" in str(exc.value) and "revert" in str(exc.value)


def test_detector_target_mismatch_raises(fitted_forecast):
    class PickyDetector(sf.ThresholdDetector):
        required_targets: ClassVar[dict] = {"revert": ("triple_barrier",)}

    det = PickyDetector(forecast="revert", p_min=0.5)
    with pytest.raises(sf.FlowConfigError):
        sf.Flow(name="bad_target", forecasts={"revert": fitted_forecast}, detectors=[det])


def test_detector_target_match_ok(fitted_forecast):
    class PickyDetector(sf.ThresholdDetector):
        required_targets: ClassVar[dict] = {"revert": ("fixed_horizon",)}

    det = PickyDetector(forecast="revert", p_min=0.5)
    flow = sf.Flow(name="ok_target", forecasts={"revert": fitted_forecast}, detectors=[det])
    assert flow.detectors[0].required_slots() == ("revert",)


def test_base_detector_requires_no_slots():
    assert sf.SmaCrossDetector(fast=3, slow=8).required_slots() == ()


def test_armed_live_without_broker_raises_flow_config(ds, fitted_forecast):
    flow = sf.Flow(
        name="armed",
        forecasts={"revert": fitted_forecast},
        detectors=[sf.ThresholdDetector(forecast="revert", p_min=0.5)],
    )
    with pytest.raises(sf.FlowConfigError):
        flow.live(ds, capital=10_000, armed=True)


def _rise_obs(*, with_validator: bool):
    import polars as pl

    from signalflow.engine.types import PortfolioSnapshot
    from signalflow.strategy.observation import Observation

    cols = {"pair": ["BTCUSDT"], "ts": [0], "signal": ["rise"]}
    if with_validator:
        cols["p_success"] = pl.Series([None], dtype=pl.Float64)
    snap = PortfolioSnapshot(
        ts=0, target="USDT", balances={"USDT": 10_000.0}, positions={}, equity=10_000.0, prices={"BTCUSDT": 100.0}
    )
    return Observation(0, pl.DataFrame(cols), snap, {})


def test_null_p_success_skipped_when_validator_present():
    strat = sf.RulesStrategy(entry=sf.Entry(size_pct=0.1, min_p_success=0.5))
    intents = strat.decide(_rise_obs(with_validator=True))
    assert not any(i.kind == sf.IntentKind.OPEN for i in intents)


def test_entry_opens_without_validator_column():
    strat = sf.RulesStrategy(entry=sf.Entry(size_pct=0.1, min_p_success=0.5))
    intents = strat.decide(_rise_obs(with_validator=False))
    assert any(i.kind == sf.IntentKind.OPEN for i in intents)


def _two_pair_rise_obs():
    import polars as pl

    from signalflow.engine.types import PortfolioSnapshot
    from signalflow.strategy.observation import Observation

    signals = pl.DataFrame({"pair": ["BTCUSDT", "ETHUSDT"], "ts": [0, 0], "signal": ["rise", "rise"]})
    snap = PortfolioSnapshot(
        ts=0,
        target="USDT",
        balances={"USDT": 10_000.0},
        positions={},
        equity=10_000.0,
        prices={"BTCUSDT": 100.0, "ETHUSDT": 50.0},
    )
    return Observation(0, signals, snap, {})


def test_per_pair_cap_opens_one_in_each_pair():
    strat = sf.RulesStrategy(entry=sf.Entry(size_pct=0.1, max_positions=10, max_positions_per_pair=1))
    opens = [i for i in strat.decide(_two_pair_rise_obs()) if i.kind == sf.IntentKind.OPEN]
    assert {i.pair for i in opens} == {"BTCUSDT", "ETHUSDT"}
    assert len(opens) == 2


def test_global_cap_limits_total_opens():
    strat = sf.RulesStrategy(entry=sf.Entry(size_pct=0.1, max_positions=1))
    opens = [i for i in strat.decide(_two_pair_rise_obs()) if i.kind == sf.IntentKind.OPEN]
    assert len(opens) == 1


def test_yaml_roundtrip(flow, ds, tmp_path):
    path = str(tmp_path / "flow.yaml")
    flow.save(path, model_dir=str(tmp_path / "models"))
    loaded = sf.Flow.load(path)
    a = flow.backtest(ds, capital=50_000).final_equity
    b = loaded.backtest(ds, capital=50_000).final_equity
    assert abs(a - b) < 1e-6
