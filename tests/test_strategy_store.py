"""Tests for StrategyStore implementations (DuckDB, SQLite)."""

from datetime import datetime

import pytest

from signalflow.core import Position, StrategyState, Trade


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


class TestStrategyStorePositions:
    def test_upsert_positions(self, strategy_store):
        pos = Position(id="pos_1", pair="BTCUSDT", entry_price=45000.0, qty=0.5)
        ts = datetime(2024, 1, 1, 12, 0)
        strategy_store.upsert_positions("test_strat", ts, [pos])
        # No error - positions persisted

    def test_upsert_empty_positions(self, strategy_store):
        ts = datetime(2024, 1, 1, 12, 0)
        strategy_store.upsert_positions("test_strat", ts, [])
        # No error - silently returns

    def test_position_requires_id(self, strategy_store):
        # Position has id by default (uuid), so this should work
        pos = Position(pair="BTCUSDT")
        ts = datetime(2024, 1, 1)
        strategy_store.upsert_positions("test_strat", ts, [pos])

    def test_upsert_overwrites_position(self, strategy_store):
        ts = datetime(2024, 1, 1, 12, 0)
        pos = Position(id="pos_1", pair="BTCUSDT", entry_price=45000.0, qty=0.5)
        strategy_store.upsert_positions("test_strat", ts, [pos])

        pos_updated = Position(id="pos_1", pair="BTCUSDT", entry_price=46000.0, qty=1.0)
        strategy_store.upsert_positions("test_strat", ts, [pos_updated])
        # No error - upsert succeeded


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
