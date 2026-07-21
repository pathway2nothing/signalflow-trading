"""Decision-loop invariants: curve reconciles with the fill log; no double-acts."""

from dataclasses import dataclass

import polars as pl
import pytest

import signalflow as sf
from signalflow.decorators import detector
from signalflow.engine.engine import Engine
from signalflow.engine.types import Fill, PortfolioSnapshot
from signalflow.enums import SIGNAL_COL, Side


@detector("d03_no_signal_col")
@dataclass
class _NoSignalDetector(sf.SignalDetector):
    """Emits a ``score`` column but never the required ``signal`` column."""

    @property
    def warmup(self) -> int:
        return 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit(1.0).alias("score"))


@detector("d03_bad_vocab")
@dataclass
class _BadVocabDetector(sf.SignalDetector):
    """Emits an off-vocabulary signal value."""

    @property
    def warmup(self) -> int:
        return 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit("BUY!!").alias(SIGNAL_COL))


@detector("d03_raises")
@dataclass
class _RaisingDetector(sf.SignalDetector):
    """Raises inside ``detect``."""

    @property
    def warmup(self) -> int:
        return 1

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        raise RuntimeError("boom in detect")


def _churny_flow() -> sf.Flow:
    return sf.Flow(
        name="inv",
        detectors=[sf.SmaCrossDetector(fast=3, slow=8)],
        strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.9), exit=sf.Exit(tp=0.005, sl=0.005)),
    )


def test_curve_reconciles_with_fill_log(ds):
    """final_equity equals a full fold of the fill log valued at last prices."""
    run = _churny_flow().backtest(ds, capital=50_000)
    last_ts = ds.frame.get_column("ts").max()
    fold_eq = Engine.fold(run.fills, 50_000.0).equity(ds.prices_at(last_ts))
    assert run.final_equity == pytest.approx(fold_eq, abs=1e-6)


def test_initial_equity_is_capital(ds):
    run = _churny_flow().backtest(ds, capital=50_000)
    assert run.initial_equity == pytest.approx(50_000.0)


def test_no_duplicate_open_on_repeated_signal():
    """Two RISE rows for one pair in one bar yield a single OPEN intent."""
    strat = sf.RulesStrategy(entry=sf.Entry(size_pct=0.1))
    snap = PortfolioSnapshot(
        ts=1,
        target="USDT",
        balances={"USDT": 10_000.0},
        positions={},
        equity=10_000.0,
        prices={"BTCUSDT": 100.0},
    )
    sig = pl.DataFrame({"pair": ["BTCUSDT", "BTCUSDT"], "ts": [1, 1], "signal": [sf.RISE, sf.RISE]})
    intents = strat.decide(sf.Observation(1, sig, snap, {}))
    opens = [i for i in intents if i.kind == sf.IntentKind.OPEN]
    assert len(opens) == 1


def test_equity_survives_missing_pair_price():
    """A held asset absent from this bar's prices does not crash valuation."""
    eng = Engine(10_000.0, target="USDT", quote="USDT")
    eng.apply([Fill(pair="BTCUSDT", ts=1, side=Side.BUY, qty=1.0, price=100.0)])
    assert eng.equity({"ETHUSDT": 3000.0}) == pytest.approx(10_000.0)


def test_detector_without_signal_column_raises(ds):
    flow = sf.Flow(name="d03a", detectors=[_NoSignalDetector()])
    with pytest.raises(sf.PipeError, match="did not produce") as exc:
        flow.backtest(ds, capital=10_000)
    assert "d03_no_signal_col" in str(exc.value)


def test_detector_off_vocabulary_raises(ds):
    flow = sf.Flow(name="d03b", detectors=[_BadVocabDetector()])
    with pytest.raises(sf.PipeError, match="invalid signal values"):
        flow.backtest(ds, capital=10_000)


def test_detector_compute_exception_wrapped(ds):
    flow = sf.Flow(name="d03c", detectors=[_RaisingDetector()])
    with pytest.raises(sf.PipeError) as exc:
        flow.backtest(ds, capital=10_000)
    assert "d03_raises" in str(exc.value)
