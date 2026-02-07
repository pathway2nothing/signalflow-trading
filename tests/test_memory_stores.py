"""Tests for InMemoryRawStore and InMemoryStrategyStore."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.data.raw_store.memory_store import InMemoryRawStore
from signalflow.data.strategy_store.memory import InMemoryStrategyStore
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade
from signalflow.core.enums import PositionType


TS = datetime(2024, 1, 1)


def _make_klines(pair="BTCUSDT", n=10, start=None):
    base = start or TS
    klines = []
    for i in range(n):
        klines.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000.0,
            }
        )
    return klines


# ── InMemoryRawStore ─────────────────────────────────────────────────────


class TestInMemoryRawStoreInsert:
    def test_insert_klines(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines())
        df = store.load("BTCUSDT")
        assert df.height == 10
        assert "pair" in df.columns
        assert "close" in df.columns

    def test_insert_empty(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", [])
        assert store.load("BTCUSDT").height == 0

    def test_upsert_overwrites(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=5))
        store.insert_klines("BTCUSDT", _make_klines(n=5))
        df = store.load("BTCUSDT")
        assert df.height == 5  # no duplicates

    def test_timezone_stripped(self):
        from datetime import timezone

        store = InMemoryRawStore()
        klines = [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        ]
        store.insert_klines("BTCUSDT", klines)
        df = store.load("BTCUSDT")
        assert df.height == 1


class TestInMemoryRawStoreQueries:
    def test_get_time_bounds(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        first, last = store.get_time_bounds("BTCUSDT")
        assert first == TS
        assert last == TS + timedelta(minutes=9)

    def test_get_time_bounds_empty(self):
        store = InMemoryRawStore()
        first, last = store.get_time_bounds("BTCUSDT")
        assert first is None
        assert last is None

    def test_find_gaps(self):
        store = InMemoryRawStore()
        # Insert 0,1,2 and 5,6,7 (gap at 3,4)
        klines = _make_klines(n=3) + _make_klines(n=3, start=TS + timedelta(minutes=5))
        store.insert_klines("BTCUSDT", klines)
        gaps = store.find_gaps("BTCUSDT", TS, TS + timedelta(minutes=7), tf_minutes=1)
        assert len(gaps) > 0

    def test_find_gaps_no_gaps(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        gaps = store.find_gaps("BTCUSDT", TS, TS + timedelta(minutes=9), tf_minutes=1)
        assert len(gaps) == 0


class TestInMemoryRawStoreLoad:
    def test_load_single_pair(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        df = store.load("BTCUSDT")
        assert df.height == 10

    def test_load_with_start_end(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        df = store.load("BTCUSDT", start=TS + timedelta(minutes=2), end=TS + timedelta(minutes=5))
        assert df.height == 4

    def test_load_with_start_only(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        df = store.load("BTCUSDT", start=TS + timedelta(minutes=5))
        assert df.height == 5

    def test_load_with_end_only(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        df = store.load("BTCUSDT", end=TS + timedelta(minutes=4))
        assert df.height == 5

    def test_load_many(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=5))
        store.insert_klines("ETHUSDT", _make_klines(n=5))
        df = store.load_many(["BTCUSDT", "ETHUSDT"])
        assert df.height == 10

    def test_load_many_empty_pairs(self):
        store = InMemoryRawStore()
        df = store.load_many([])
        assert df.height == 0

    def test_load_many_with_start_end(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        df = store.load_many(["BTCUSDT"], start=TS + timedelta(minutes=2), end=TS + timedelta(minutes=5))
        assert df.height == 4

    def test_load_many_pandas(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=5))
        pdf = store.load_many_pandas(["BTCUSDT"])
        assert len(pdf) == 5
        assert "pair" in pdf.columns


class TestInMemoryRawStoreStats:
    def test_get_stats(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        stats = store.get_stats()
        assert stats.height == 1
        assert stats["pair"][0] == "BTCUSDT"
        assert stats["rows"][0] == 10

    def test_get_stats_empty(self):
        store = InMemoryRawStore()
        stats = store.get_stats()
        assert stats.height == 0

    def test_close(self):
        store = InMemoryRawStore()
        store.insert_klines("BTCUSDT", _make_klines(n=10))
        store.close()
        assert store.load("BTCUSDT").height == 0


# ── InMemoryStrategyStore ────────────────────────────────────────────────


class TestInMemoryStrategyStore:
    def test_init(self):
        store = InMemoryStrategyStore()
        store.init()  # should not raise

    def test_save_and_load_state(self):
        store = InMemoryStrategyStore()
        state = StrategyState(strategy_id="s1")
        state.portfolio.cash = 5000.0
        store.save_state(state)
        loaded = store.load_state("s1")
        assert loaded is not None
        assert loaded.strategy_id == "s1"
        # After deserialization, portfolio may be dict
        if hasattr(loaded.portfolio, "cash"):
            assert loaded.portfolio.cash == pytest.approx(5000.0)
        else:
            assert loaded.portfolio["cash"] == pytest.approx(5000.0)

    def test_load_state_not_found(self):
        store = InMemoryStrategyStore()
        assert store.load_state("nonexistent") is None

    def test_upsert_positions(self):
        store = InMemoryStrategyStore()
        pos = Position(id="p1", pair="BTCUSDT", position_type=PositionType.LONG, entry_price=100.0, qty=1.0)
        store.upsert_positions("s1", TS, [pos])
        assert ("s1", TS, "p1") in store._positions

    def test_upsert_position_no_id_raises(self):
        store = InMemoryStrategyStore()

        class FakePos:
            id = None

        with pytest.raises(ValueError, match="Position must have id"):
            store.upsert_positions("s1", TS, [FakePos()])

    def test_append_trade(self):
        store = InMemoryStrategyStore()
        trade = Trade(id="t1", position_id="p1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        store.append_trade("s1", trade)
        assert ("s1", "t1") in store._trades

    def test_append_trade_duplicate_ignored(self):
        store = InMemoryStrategyStore()
        trade = Trade(id="t1", position_id="p1", pair="BTCUSDT", side="BUY", ts=TS, price=100.0, qty=1.0, fee=0.1)
        store.append_trade("s1", trade)
        store.append_trade("s1", trade)  # no error, just ignored
        assert len([k for k in store._trades if k[0] == "s1"]) == 1

    def test_append_metrics(self):
        store = InMemoryStrategyStore()
        store.append_metrics("s1", TS, {"equity": 10000.0, "drawdown": 0.05})
        assert store._metrics[("s1", TS, "equity")] == pytest.approx(10000.0)
        assert store._metrics[("s1", TS, "drawdown")] == pytest.approx(0.05)

    def test_close(self):
        store = InMemoryStrategyStore()
        state = StrategyState(strategy_id="s1")
        store.save_state(state)
        store.close()
        assert store.load_state("s1") is None
