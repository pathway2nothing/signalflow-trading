"""End-to-end Flow: backtest, leakage invariant, yaml round-trip, quicktest."""


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
    assert sc["promotable"] is True
    assert run.equity_curve.height > 0
    assert sc["initial_equity"] == 50_000.0


def test_quicktest_not_promotable(flow, ds):
    run = flow.quicktest(ds, capital=50_000)
    assert run.promotable is False


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


def test_yaml_roundtrip(flow, ds, tmp_path):
    path = str(tmp_path / "flow.yaml")
    flow.save(path, model_dir=str(tmp_path / "models"))
    loaded = sf.Flow.load(path)
    a = flow.backtest(ds, capital=50_000).final_equity
    b = loaded.backtest(ds, capital=50_000).final_equity
    assert abs(a - b) < 1e-6
