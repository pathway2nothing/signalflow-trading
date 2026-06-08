"""Tests for StrategyStore implementations (DuckDB, SQLite)."""

from datetime import datetime

import pytest

from signalflow.core import CashPolicy, StrategyState, Trade, apply_fill


class TestStrategyStoreSaveLoad:
    def test_load_missing_returns_none(self, strategy_store):
        assert strategy_store.load_state("nonexistent") is None

    def test_save_and_load_state(self, strategy_store):
        state = StrategyState(strategy_id="test_strat")
        state.portfolio.cash = 10000.0
        strategy_store.save_state(state)

        loaded = strategy_store.load_state("test_strat")
        assert loaded is not None
        assert loaded.strategy_id == "test_strat"

    def test_save_overwrites(self, strategy_store):
        state = StrategyState(strategy_id="test_strat")
        state.portfolio.cash = 10000.0
        strategy_store.save_state(state)

        state.portfolio.cash = 5000.0
        state.last_ts = datetime(2024, 6, 1)
        strategy_store.save_state(state)

        loaded = strategy_store.load_state("test_strat")
        assert loaded is not None

    def test_multiple_strategies(self, strategy_store):
        for sid in ["strat_a", "strat_b", "strat_c"]:
            state = StrategyState(strategy_id=sid)
            strategy_store.save_state(state)

        for sid in ["strat_a", "strat_b", "strat_c"]:
            loaded = strategy_store.load_state(sid)
            assert loaded is not None
            assert loaded.strategy_id == sid


class TestStrategyStoreReadTrades:
    def test_read_trades_empty(self, strategy_store):
        assert strategy_store.read_trades("nonexistent") == []

    def test_read_trades_round_trip(self, strategy_store):
        t1 = Trade(
            id="t1",
            position_id="p1",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime(2024, 1, 1, 10),
            price=100.0,
            qty=1.0,
            fee=0.1,
            meta={"type": "entry", "signal_strength": 0.8},
        )
        t2 = Trade(
            id="t2",
            position_id="p1",
            pair="BTCUSDT",
            side="SELL",
            ts=datetime(2024, 1, 1, 12),
            price=110.0,
            qty=1.0,
            fee=0.11,
            meta={"type": "exit"},
        )
        strategy_store.append_trade("s", t1)
        strategy_store.append_trade("s", t2)

        trades = strategy_store.read_trades("s")
        assert [t.id for t in trades] == ["t1", "t2"]  # chronological
        assert trades[0].ts == datetime(2024, 1, 1, 10)
        assert trades[0].meta["signal_strength"] == 0.8
        assert trades[1].side == "SELL"

    def test_verify_snapshot_matches_replay(self, strategy_store):
        # Build a portfolio via the event log, append the trades, save the snapshot.
        state = StrategyState(strategy_id="s")
        state.portfolio.cash = 10_000.0
        entry = Trade(
            id="t1",
            position_id="p1",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime(2024, 1, 1, 10),
            price=100.0,
            qty=1.0,
            fee=0.1,
            meta={"type": "entry"},
        )
        apply_fill(state.portfolio, entry, policy=CashPolicy())
        strategy_store.append_trade("s", entry)
        strategy_store.save_state(state)

        assert strategy_store.verify_snapshot("s", initial_cash=10_000.0) is True

    def test_verify_snapshot_detects_drift(self, strategy_store):
        state = StrategyState(strategy_id="s")
        state.portfolio.cash = 10_000.0
        entry = Trade(
            id="t1",
            position_id="p1",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime(2024, 1, 1, 10),
            price=100.0,
            qty=1.0,
            fee=0.1,
            meta={"type": "entry"},
        )
        apply_fill(state.portfolio, entry, policy=CashPolicy())
        # Tamper the snapshot cash so it no longer matches the log replay.
        state.portfolio.cash += 999.0
        strategy_store.append_trade("s", entry)
        strategy_store.save_state(state)

        assert strategy_store.verify_snapshot("s", initial_cash=10_000.0) is False

    def test_verify_snapshot_no_state(self, strategy_store):
        assert strategy_store.verify_snapshot("never_saved") is False


class TestStrategyStoreTrades:
    def test_append_trade(self, strategy_store):
        trade = Trade(
            id="trade_1",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime(2024, 1, 1, 12, 0),
            price=45000.0,
            qty=0.5,
            fee=22.5,
        )
        strategy_store.append_trade("test_strat", trade)
        # No error - trade persisted

    def test_append_trade_idempotent(self, strategy_store):
        trade = Trade(
            id="trade_1",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime(2024, 1, 1, 12, 0),
            price=45000.0,
            qty=0.5,
        )
        strategy_store.append_trade("test_strat", trade)
        strategy_store.append_trade("test_strat", trade)  # duplicate - should be ignored

    def test_trade_requires_id_and_ts(self, strategy_store):
        # Trade() has default id (uuid) and ts=None
        trade = Trade()
        with pytest.raises(ValueError, match="Trade must have id and ts"):
            strategy_store.append_trade("test_strat", trade)


class TestStrategyStoreMetrics:
    def test_append_metrics(self, strategy_store):
        ts = datetime(2024, 1, 1, 12, 0)
        metrics = {"total_return": 0.05, "sharpe_ratio": 1.2, "max_drawdown": -0.03}
        strategy_store.append_metrics("test_strat", ts, metrics)
        # No error - metrics persisted

    def test_append_empty_metrics(self, strategy_store):
        ts = datetime(2024, 1, 1, 12, 0)
        strategy_store.append_metrics("test_strat", ts, {})
        # No error - silently returns

    def test_metrics_upsert_on_conflict(self, strategy_store):
        ts = datetime(2024, 1, 1, 12, 0)
        strategy_store.append_metrics("test_strat", ts, {"sharpe": 1.0})
        strategy_store.append_metrics("test_strat", ts, {"sharpe": 2.0})  # update
        # No error - upsert succeeded
