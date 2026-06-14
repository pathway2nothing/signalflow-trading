"""Decision-loop invariants: curve reconciles with the fill log; no double-acts."""

import polars as pl
import pytest

import signalflow as sf
from signalflow.engine.engine import Engine
from signalflow.engine.types import Fill, PortfolioSnapshot
from signalflow.enums import Side


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
        ts=1, target="USDT", balances={"USDT": 10_000.0}, positions={}, equity=10_000.0,
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
