"""Additive warmup composition and the warmup canary."""

from dataclasses import dataclass

import polars as pl
import pytest

import signalflow as sf
from signalflow.transform.base import Feature
from signalflow.transform.warmup import check_flow, check_pipeline, measure_warmup


@dataclass
class _RollingOf(Feature):
    """Rolling mean of an arbitrary column; ``declared`` overrides the honest warmup."""

    col: str = "close"
    length: int = 30
    declared: int | None = None

    @property
    def warmup(self) -> int:
        return self.length if self.declared is None else self.declared

    @property
    def requires(self) -> list[str]:
        return [self.col]

    @property
    def outputs(self) -> list[str]:
        return [f"{self.col}_rm{self.length}"]

    def exprs(self) -> list[pl.Expr]:
        return [pl.col(self.col).rolling_mean(self.length).alias(self.outputs[0])]


@dataclass
class _Opaque(Feature):
    """A feature that does not declare what it reads (``requires`` is None)."""

    @property
    def warmup(self) -> int:
        return 10

    @property
    def outputs(self) -> list[str]:
        return ["opaque"]

    def exprs(self) -> list[pl.Expr]:
        return [pl.col("close").rolling_mean(10).alias("opaque")]


class _UnderDetector(sf.SignalDetector):
    """Needs 30 bars, declares 5."""

    @property
    def warmup(self) -> int:
        return 5

    def detect(self, df):
        sma = pl.col("close").rolling_mean(30).over("pair")
        return df.with_columns(
            pl.when(pl.col("close") > sma).then(pl.lit(sf.RISE)).otherwise(pl.lit(sf.NONE)).alias("signal")
        )


# --- part 1: additive composition -------------------------------------------------


def test_independent_steps_take_the_max():
    pipe = sf.FeaturePipeline(sf.SMA(20), sf.SMA(50))
    assert pipe.effective_warmups() == [20, 50]
    assert pipe.warmup == 50


def test_chained_steps_add_their_producers_warmup():
    pipe = sf.FeaturePipeline(sf.SMA(20), _RollingOf("sma_20", 30))
    assert pipe.effective_warmups() == [20, 50]
    assert pipe.warmup == 50


def test_unknown_requires_is_charged_the_largest_warmup_so_far():
    pipe = sf.FeaturePipeline(sf.SMA(50), sf.SMA(20), _Opaque())
    assert pipe.effective_warmups() == [50, 20, 60]


def test_stateful_tail_inherits_its_inputs():
    pipe = sf.FeaturePipeline(sf.SMA(20), sf.SMA(50), sf.WoE(), sf.IVSelector())
    assert pipe.effective_warmups() == [20, 50, 50, 50]
    assert pipe.warmup == 50


# --- part 2: the canary -----------------------------------------------------------


def test_measure_sma_is_exact():
    check = measure_warmup(sf.SMA(20), exact=True)
    assert check.ok and check.measured == 20


def test_measure_catches_an_under_declared_feature():
    check = measure_warmup(_RollingOf("close", 30, declared=5))
    assert not check.ok
    assert check.measured == 30
    assert "under-declared by 25" in check.detail


def test_check_pipeline_measures_effective_need_of_a_chain():
    honest = check_pipeline(sf.FeaturePipeline(sf.SMA(20), _RollingOf("sma_20", 30)), exact=True)
    assert [c.ok for c in honest] == [True, True]
    assert honest[1].declared == 50 and honest[1].measured == 49  # the chain needs 20 + 30 - 1 bars

    lying = check_pipeline(sf.FeaturePipeline(sf.SMA(20), _RollingOf("sma_20", 30, declared=3)))
    assert not lying[1].ok and lying[1].declared == 23 and lying[1].measured == 49


def test_check_pipeline_skips_stateful_steps():
    checks = check_pipeline(sf.FeaturePipeline(sf.SMA(20), sf.WoE()))
    assert checks[1].ok and "stateful" in checks[1].detail


def test_check_flow_passes_for_honest_components():
    flow = sf.Flow(name="ok", detectors=[sf.SmaCrossDetector(fast=3, slow=59)], strategy=sf.RulesStrategy())
    checks = flow.check_warmup()
    assert all(c.ok for c in checks)


def test_check_flow_reports_forecast_readers_via_the_model(ds):
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipeline(sf.SMA(20), _RollingOf("sma_20", 30)),
        output="p_rise",
        cv=sf.KFold(3),
    ).fit(ds)
    flow = sf.Flow(
        name="m",
        forecasts={"rise": model},
        detectors=[sf.ThresholdDetector(forecast="rise", p_min=0.6)],
        strategy=sf.RulesStrategy(),
    )
    checks = check_flow(flow)
    names = [c.component for c in checks]
    covered = [c for n, c in zip(names, checks, strict=True) if n.startswith("detector")]
    assert covered and all("covered by the model" in c.detail for c in covered)
    assert any(n.startswith("forecast rise/") for n in names)
    assert all(c.ok for c in checks)
    assert flow.required_warmup == 50


def test_under_declared_detector_fails_the_canary_and_blocks_simulate(ds):
    flow = sf.Flow(name="under", detectors=[_UnderDetector()], strategy=sf.RulesStrategy())
    checks = flow.check_warmup(raise_on_fail=False)
    bad = [c for c in checks if not c.ok]
    assert bad and bad[0].measured is not None and bad[0].measured > 5
    with pytest.raises(sf.WarmupError):
        flow.check_warmup()
    with pytest.raises(sf.WarmupError):
        flow.simulate(ds, capital=10_000)
    run = flow.simulate(ds, capital=10_000, check_warmup=False)
    assert run.equity_curve.height > 0
