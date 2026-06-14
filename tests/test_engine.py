"""Engine: event-sourced state and the single-position-math invariant."""


import pytest

from signalflow.engine import Engine, Fill
from signalflow.enums import Side


def _fills():
    return [
        Fill("BTCUSDT", ts=1, side=Side.BUY, qty=0.5, price=100.0, fee=0.05),
        Fill("BTCUSDT", ts=2, side=Side.BUY, qty=0.5, price=120.0, fee=0.06),
        Fill("ETHUSDT", ts=3, side=Side.BUY, qty=2.0, price=50.0, fee=0.10),
        Fill("BTCUSDT", ts=4, side=Side.SELL, qty=0.4, price=130.0, fee=0.05),
    ]


def test_apply_equals_fold():
    """Incremental apply must equal a full fold of the event log (one impl)."""
    fills = _fills()
    eng = Engine(capital={"USDT": 1000.0}, target="USDT")
    eng.apply(fills)
    folded = Engine.fold(fills, capital={"USDT": 1000.0}, target="USDT")
    assert eng.balances == folded.balances
    assert {k: vars(v) for k, v in eng.positions.items()} == {k: vars(v) for k, v in folded.positions.items()}


def test_avg_price_and_equity():
    eng = Engine(capital={"USDT": 1000.0}, target="USDT")
    eng.apply(_fills()[:2])
    pos = eng.positions["BTCUSDT"]
    assert pos.qty == pytest.approx(1.0)
    assert pos.avg_price == pytest.approx(110.0)
    eq = eng.equity({"BTCUSDT": 120.0})

    assert eq > 0


def test_close_clears_position():
    eng = Engine(1000.0, target="USDT")
    eng.apply([Fill("BTCUSDT", 1, Side.BUY, 1.0, 100.0), Fill("BTCUSDT", 2, Side.SELL, 1.0, 110.0)])
    assert "BTCUSDT" not in eng.positions
