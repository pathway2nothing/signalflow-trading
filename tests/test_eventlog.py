"""Tests for the event log (core/eventlog.py): apply_fill / fold / replay."""

from datetime import datetime, timedelta

from signalflow.core import (
    CashPolicy,
    Portfolio,
    PositionType,
    Trade,
    apply_fill,
    fold,
    portfolios_match,
    replay_state,
)


def _entry(pid, side, price, qty, fee, ts, ss=1.0):
    return Trade(
        id=f"{pid}-entry",
        position_id=pid,
        pair="BTCUSDT",
        side=side,
        ts=ts,
        price=price,
        qty=qty,
        fee=fee,
        meta={"type": "entry", "signal_strength": ss},
    )


def _exit(pid, side, price, qty, fee, ts):
    return Trade(
        id=f"{pid}-exit",
        position_id=pid,
        pair="BTCUSDT",
        side=side,
        ts=ts,
        price=price,
        qty=qty,
        fee=fee,
        meta={"type": "exit"},
    )


class TestApplyFill:
    def test_entry_creates_position(self):
        pf = Portfolio(cash=1000.0)
        ts = datetime(2024, 1, 1, 10)
        pos = apply_fill(pf, _entry("p1", "BUY", 100.0, 1.0, 0.1, ts, ss=0.8))

        assert pos.id == "p1"
        assert pf.positions["p1"].qty == 1.0
        assert pf.positions["p1"].position_type == PositionType.LONG
        assert pf.positions["p1"].signal_strength == 0.8
        # BUY debits notional + fee
        assert pf.cash == 1000.0 - 100.0 - 0.1

    def test_exit_closes_position_and_credits_cash(self):
        pf = Portfolio(cash=1000.0)
        ts = datetime(2024, 1, 1, 10)
        apply_fill(pf, _entry("p1", "BUY", 100.0, 1.0, 0.1, ts))
        apply_fill(pf, _exit("p1", "SELL", 110.0, 1.0, 0.11, ts + timedelta(hours=1)))

        pos = pf.positions["p1"]
        assert pos.is_closed
        assert pos.qty == 0.0
        assert abs(pos.realized_pnl - 10.0) < 1e-9  # (110-100)*1
        # cash: 1000 -100.1 (buy) +110 -0.11 (sell)
        assert abs(pf.cash - (1000.0 - 100.1 + 110.0 - 0.11)) < 1e-9

    def test_unlimited_policy_leaves_cash_untouched(self):
        pf = Portfolio(cash=1000.0)
        ts = datetime(2024, 1, 1, 10)
        apply_fill(pf, _entry("p1", "BUY", 100.0, 1.0, 0.1, ts), policy=CashPolicy(track_cash=False))
        assert pf.cash == 1000.0
        assert pf.positions["p1"].qty == 1.0


class TestFoldAndReplay:
    def test_fold_matches_incremental_apply(self):
        ts = datetime(2024, 1, 1, 10)
        events = [
            _entry("p1", "BUY", 100.0, 1.0, 0.1, ts),
            _entry("p2", "SELL", 50.0, 2.0, 0.05, ts + timedelta(hours=1)),
            _exit("p1", "SELL", 110.0, 1.0, 0.11, ts + timedelta(hours=2)),
        ]
        # incremental
        incr = Portfolio(cash=10_000.0)
        for ev in events:
            apply_fill(incr, ev)
        # fold
        folded = fold(events, initial_cash=10_000.0)

        assert portfolios_match(incr, folded)

    def test_replay_state_reconstructs_portfolio(self):
        ts = datetime(2024, 1, 1, 10)
        events = [_entry("p1", "BUY", 100.0, 1.0, 0.1, ts)]
        state = replay_state("strat", events, initial_cash=500.0)
        assert state.strategy_id == "strat"
        assert state.portfolio.positions["p1"].qty == 1.0
        assert state.portfolio.cash == 500.0 - 100.1

    def test_portfolios_match_ignores_marks(self):
        ts = datetime(2024, 1, 1, 10)
        events = [_entry("p1", "BUY", 100.0, 1.0, 0.1, ts)]
        a = fold(events, initial_cash=1000.0)
        b = fold(events, initial_cash=1000.0)
        # mark one side to a different price — should NOT affect equality
        b.positions["p1"].mark(ts=ts + timedelta(hours=5), price=999.0)
        assert portfolios_match(a, b)

    def test_portfolios_match_detects_qty_divergence(self):
        ts = datetime(2024, 1, 1, 10)
        a = fold([_entry("p1", "BUY", 100.0, 1.0, 0.1, ts)], initial_cash=1000.0)
        b = fold([_entry("p1", "BUY", 100.0, 2.0, 0.1, ts)], initial_cash=1000.0)
        assert not portfolios_match(a, b)
