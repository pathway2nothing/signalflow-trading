"""Tests for PostgreSQL store implementations.

Skipped entirely if SIGNALFLOW_PG_DSN is not set.
"""

import os
from datetime import datetime, timedelta

import pytest

from signalflow.core import Position, StrategyState, Trade

PG_DSN = os.environ.get("SIGNALFLOW_PG_DSN", "")
pg_available = pytest.mark.skipif(not PG_DSN, reason="SIGNALFLOW_PG_DSN not set")


@pg_available
class TestPgSpotStore:
    @pytest.fixture(autouse=True)
    def pg_store(self):
        from signalflow.data.raw_store.pg_stores import PgSpotStore

        store = PgSpotStore(dsn=PG_DSN, timeframe="1m")
        yield store
        # Cleanup tables between tests
        with store._con.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ohlcv, meta")
        store._con.commit()
        store.close()

    @pytest.fixture
    def klines(self):
        base = datetime(2024, 1, 1)
        return [
            {
                "timestamp": base + timedelta(minutes=i),
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 1000.0 + i * 10,
                "trades": 50 + i,
            }
            for i in range(50)
        ]

    def test_insert_and_load(self, pg_store, klines):
        pg_store.insert_klines("BTCUSDT", klines)
        df = pg_store.load("BTCUSDT")
        assert len(df) == 50

    def test_load_many(self, pg_store, klines):
        pg_store.insert_klines("BTCUSDT", klines)
        pg_store.insert_klines("ETHUSDT", klines[:10])
        df = pg_store.load_many(["BTCUSDT", "ETHUSDT"])
        assert len(df) == 60

    def test_load_with_time_range(self, pg_store, klines):
        pg_store.insert_klines("BTCUSDT", klines)
        start = datetime(2024, 1, 1, 0, 10)
        end = datetime(2024, 1, 1, 0, 19)
        df = pg_store.load("BTCUSDT", start=start, end=end)
        assert len(df) == 10

    def test_get_time_bounds(self, pg_store, klines):
        pg_store.insert_klines("BTCUSDT", klines)
        mn, mx = pg_store.get_time_bounds("BTCUSDT")
        assert mn == datetime(2024, 1, 1)
        assert mx == datetime(2024, 1, 1, 0, 49)

    def test_get_stats(self, pg_store, klines):
        pg_store.insert_klines("BTCUSDT", klines)
        stats = pg_store.get_stats()
        assert len(stats) == 1


@pg_available
class TestPgStrategyStore:
    @pytest.fixture(autouse=True)
    def pg_store(self):
        from signalflow.data.strategy_store.pg import PgStrategyStore

        store = PgStrategyStore(dsn=PG_DSN)
        store.init()
        yield store
        # Cleanup
        with store.con.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS strategy_state, positions, trades, metrics")
        store.con.commit()
        store.close()

    def test_save_and_load_state(self, pg_store):
        state = StrategyState(strategy_id="test")
        state.portfolio.cash = 10000.0
        pg_store.save_state(state)

        loaded = pg_store.load_state("test")
        assert loaded is not None
        assert loaded.strategy_id == "test"

    def test_load_missing_returns_none(self, pg_store):
        assert pg_store.load_state("nope") is None

    def test_append_trade(self, pg_store):
        trade = Trade(
            id="t1",
            pair="BTCUSDT",
            side="BUY",
            ts=datetime(2024, 1, 1),
            price=45000.0,
            qty=0.5,
        )
        pg_store.append_trade("test", trade)

    def test_append_metrics(self, pg_store):
        pg_store.append_metrics("test", datetime(2024, 1, 1), {"sharpe": 1.5})

    def test_upsert_positions(self, pg_store):
        pos = Position(id="p1", pair="BTCUSDT", entry_price=45000.0, qty=0.5)
        pg_store.upsert_positions("test", datetime(2024, 1, 1), [pos])
