"""SF_VERBOSE coverage: the important steps emit DEBUG detail and one INFO summary each."""

import pytest
from loguru import logger

import signalflow as sf
from signalflow._logging import step


@pytest.fixture
def lines():
    captured: list[tuple[str, str]] = []
    sink = logger.add(lambda m: captured.append((m.record["level"].name, m.record["message"])), level="TRACE")
    yield captured
    logger.remove(sink)


def _messages(lines, level=None):
    return [msg for lvl, msg in lines if level is None or lvl == level]


def test_fit_and_backtest_report_each_step(lines):
    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2023-01-01", end="2023-02-15", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipeline(sf.SMA(10), sf.SMA(20), sf.WoE(), sf.IVSelector()),
    )
    model.fit(ds)
    flow = sf.Flow(
        name="log_probe",
        forecasts={"rise": model},
        detectors=[sf.ThresholdDetector(forecast="rise", p_min=0.5)],
        strategy=sf.RulesStrategy(),
    )
    flow.backtest(ds, capital=10_000)

    debug = "\n".join(_messages(lines, "DEBUG"))
    for needle in (
        "Dataset.from_source: source=synthetic",
        "FeaturePipeline: fused 2 features",
        "FeaturePipeline.compute: 2 transforms",
        "ForecastModel.fit(p_rise): backend=lightgbm",
        "ForecastModel.fit: features",
        "ForecastModel.fit: sampler",
        "ForecastModel.fit: labels",
        "fold 1/",
        "fitted kept=",
        "ForecastModel.fit: final production stack",
        "ForecastModel.predict(p_rise)",
        "Flow.backtest('log_probe'): capital=10000",
        "forecast slot 'rise': in-sample rows=",
        "detector 'threshold'",
        "Flow.backtest: loop bars=",
    ):
        assert needle in debug, f"missing DEBUG line containing {needle!r}"

    trace = "\n".join(_messages(lines, "TRACE"))
    assert "WoE.fit:" in trace and "IVSelector.fit: kept" in trace, "encoder internals belong to TRACE"
    assert "WoE.fit:" not in debug

    info = _messages(lines, "INFO")
    assert sum(m.startswith("ForecastModel.fit(p_rise):") for m in info) == 1
    assert sum(m.startswith("Flow.backtest('log_probe'):") for m in info) == 1
    summary = next(m for m in info if m.startswith("Flow.backtest("))
    assert "bars=1,080" in summary and "equity" in summary and "max_dd" in summary


def test_simulate_logs_progress_not_per_bar_detail(lines):
    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2023-01-01", end="2023-01-08", interval="1h")
    flow = sf.Flow(name="sim_probe", detectors=[sf.SmaCrossDetector(fast=5, slow=10)])
    flow.simulate(ds, capital=10_000)

    debug = _messages(lines, "DEBUG")
    assert not [m for m in debug if m.startswith("detector '")], "live loop must not log detectors per bar"
    assert any(m.startswith("Flow.simulate('sim_probe'): warmup bars=") for m in debug)
    assert sum(m.startswith("Flow.simulate('sim_probe'):") for m in _messages(lines, "INFO")) == 1


def test_step_reports_failure_and_reraises(lines):
    with pytest.raises(ValueError, match="boom"), step("probe", k=1):
        raise ValueError("boom")
    assert any(m.startswith("probe: failed after") and "ValueError: boom" in m for m in _messages(lines, "DEBUG"))

    with step("probe", k=1) as log:
        log["rows"] = 3
    assert any(m.startswith("probe: k=1 rows=3 (") for m in _messages(lines, "DEBUG"))
