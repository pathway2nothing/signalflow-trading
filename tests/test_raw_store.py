"""Tests for RawDataStore implementations (DuckDB, SQLite)."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


class TestRawStoreInsertAndLoad:
    def test_insert_and_load_single_pair(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        df = raw_store.load("BTCUSDT")
        assert len(df) == 100
        assert df["pair"][0] == "BTCUSDT"

    def test_insert_and_load_many(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        raw_store.insert_klines("ETHUSDT", sample_klines)
        df = raw_store.load_many(["BTCUSDT", "ETHUSDT"])
        assert len(df) == 200
        assert set(df["pair"].unique().to_list()) == {"BTCUSDT", "ETHUSDT"}

    def test_load_with_time_range(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        start = datetime(2024, 1, 1, 0, 10)
        end = datetime(2024, 1, 1, 0, 19)
        df = raw_store.load("BTCUSDT", start=start, end=end)
        assert len(df) == 10
        assert df["timestamp"].min() >= start
        assert df["timestamp"].max() <= end

    def test_load_empty_pairs_returns_empty_df(self, raw_store):
        df = raw_store.load_many([])
        assert len(df) == 0
        assert "pair" in df.columns
        assert "timestamp" in df.columns

    def test_load_many_pandas(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        df = raw_store.load_many_pandas(["BTCUSDT"])
        assert len(df) == 100
        assert "pair" in df.columns

    def test_upsert_overwrites_existing(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)

        # Update first kline with different price
        updated = [
            {
                "timestamp": datetime(2024, 1, 1, 0, 0),
                "open": 999.0,
                "high": 999.0,
                "low": 999.0,
                "close": 999.0,
                "volume": 999.0,
                "trades": 999,
            }
        ]
        raw_store.insert_klines("BTCUSDT", updated)

        df = raw_store.load("BTCUSDT")
        assert len(df) == 100  # still 100 rows
        first = df.sort("timestamp").row(0, named=True)
        assert first["open"] == 999.0

    def test_insert_empty_klines(self, raw_store):
        raw_store.insert_klines("BTCUSDT", [])
        df = raw_store.load("BTCUSDT")
        assert len(df) == 0


class TestRawStoreQueries:
    def test_get_time_bounds(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        mn, mx = raw_store.get_time_bounds("BTCUSDT")
        assert mn == datetime(2024, 1, 1, 0, 0)
        assert mx == datetime(2024, 1, 1, 1, 39)

    def test_get_time_bounds_empty(self, raw_store):
        mn, mx = raw_store.get_time_bounds("NONEXISTENT")
        assert mn is None
        assert mx is None

    def test_find_gaps(self, raw_store):
        base = datetime(2024, 1, 1)
        # Insert bars 0-4 and 10-14, leaving a gap at 5-9
        klines_a = [
            {
                "timestamp": base + timedelta(minutes=i),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
            for i in range(5)
        ]
        klines_b = [
            {
                "timestamp": base + timedelta(minutes=i),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
            for i in range(10, 15)
        ]
        raw_store.insert_klines("BTCUSDT", klines_a)
        raw_store.insert_klines("BTCUSDT", klines_b)

        gaps = raw_store.find_gaps(
            "BTCUSDT",
            start=base,
            end=base + timedelta(minutes=14),
            tf_minutes=1,
        )
        assert len(gaps) == 1
        assert gaps[0][0] == base + timedelta(minutes=5)

    def test_get_stats(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        raw_store.insert_klines("ETHUSDT", sample_klines[:10])
        stats = raw_store.get_stats()
        assert len(stats) == 2
        assert "pair" in stats.columns
        btc_row = stats.filter(pl.col("pair") == "BTCUSDT").row(0, named=True)
        assert btc_row["rows"] == 100


class TestRawStoreSchema:
    def test_schema_columns(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        df = raw_store.load("BTCUSDT")
        expected = {"pair", "timestamp", "open", "high", "low", "close", "volume", "trades"}
        assert set(df.columns) == expected

    def test_timestamp_is_timezone_naive(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        df = raw_store.load("BTCUSDT")
        ts_dtype = df.schema["timestamp"]
        # Should be Datetime without timezone
        assert ts_dtype == pl.Datetime or (hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone is None)

    def test_load_with_start_only(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        start = datetime(2024, 1, 1, 1, 0)
        df = raw_store.load("BTCUSDT", start=start)
        assert all(ts >= start for ts in df["timestamp"].to_list())

    def test_load_with_end_only(self, raw_store, sample_klines):
        raw_store.insert_klines("BTCUSDT", sample_klines)
        end = datetime(2024, 1, 1, 0, 10)
        df = raw_store.load("BTCUSDT", end=end)
        assert all(ts <= end for ts in df["timestamp"].to_list())


class TestRawStoreToRawData:
    """Tests for RawDataStore.to_raw_data() method."""

    def test_to_raw_data_returns_raw_data(self, raw_store, sample_klines):
        """to_raw_data returns RawData container with correct data."""
        from signalflow.core import RawData

        raw_store.insert_klines("BTCUSDT", sample_klines)
        raw_store.insert_klines("ETHUSDT", sample_klines[:50])

        start = datetime(2024, 1, 1, 0, 0)
        end = datetime(2024, 1, 1, 1, 39)

        raw_data = raw_store.to_raw_data(
            pairs=["BTCUSDT", "ETHUSDT"],
            start=start,
            end=end,
        )

        assert isinstance(raw_data, RawData)
        assert raw_data.datetime_start == start
        assert raw_data.datetime_end == end
        assert raw_data.pairs == ["BTCUSDT", "ETHUSDT"]
        assert "spot" in raw_data.data

    def test_to_raw_data_custom_key(self, raw_store, sample_klines):
        """to_raw_data with custom data_key."""
        raw_store.insert_klines("BTCUSDT", sample_klines)

        raw_data = raw_store.to_raw_data(
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 1, 39),
            data_key="custom_key",
        )

        assert "custom_key" in raw_data.data
        assert "spot" not in raw_data.data

    def test_to_raw_data_sorted_by_pair_timestamp(self, raw_store, sample_klines):
        """to_raw_data returns data sorted by (pair, timestamp)."""
        raw_store.insert_klines("BTCUSDT", sample_klines)
        raw_store.insert_klines("ETHUSDT", sample_klines)

        raw_data = raw_store.to_raw_data(
            pairs=["BTCUSDT", "ETHUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 1, 39),
        )

        df = raw_data["spot"]
        # Check sorted
        pairs_sorted = df["pair"].to_list()
        btc_count = pairs_sorted.count("BTCUSDT")
        assert all(p == "BTCUSDT" for p in pairs_sorted[:btc_count])
        assert all(p == "ETHUSDT" for p in pairs_sorted[btc_count:])

    def test_to_raw_data_empty(self, raw_store):
        """to_raw_data with no data returns empty DataFrame in RawData."""
        raw_data = raw_store.to_raw_data(
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )

        assert "spot" in raw_data.data
        assert len(raw_data["spot"]) == 0


class TestDuckDbRawStoreDataTypes:
    """Tests for DuckDbRawStore data_type support (futures, perpetual)."""

    def _make_kline(self, minute: int, **extra: float) -> dict:
        base = datetime(2024, 1, 1)
        k: dict = {
            "timestamp": base + timedelta(minutes=minute),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
            "trades": 50,
        }
        k.update(extra)
        return k

    def test_futures_insert_and_load(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "futures.duckdb", data_type="futures")
        klines = [self._make_kline(i, open_interest=50000.0 + i) for i in range(20)]
        store.insert_klines("BTCUSDT", klines)

        df = store.load("BTCUSDT")
        assert len(df) == 20
        assert "open_interest" in df.columns
        assert df["open_interest"][0] == pytest.approx(50000.0)
        store.close()

    def test_perpetual_insert_and_load(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "perp.duckdb", data_type="perpetual")
        klines = [self._make_kline(i, funding_rate=0.0001, open_interest=50000.0 + i) for i in range(20)]
        store.insert_klines("BTCUSDT", klines)

        df = store.load("BTCUSDT")
        assert len(df) == 20
        assert "funding_rate" in df.columns
        assert "open_interest" in df.columns
        assert df["funding_rate"][0] == pytest.approx(0.0001)
        store.close()

    def test_futures_schema_columns(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "f.duckdb", data_type="futures")
        klines = [self._make_kline(0, open_interest=50000.0)]
        store.insert_klines("BTCUSDT", klines)

        df = store.load("BTCUSDT")
        expected = {"pair", "timestamp", "open", "high", "low", "close", "volume", "open_interest", "trades"}
        assert set(df.columns) == expected
        store.close()

    def test_perpetual_schema_columns(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "p.duckdb", data_type="perpetual")
        klines = [self._make_kline(0, funding_rate=0.0001, open_interest=50000.0)]
        store.insert_klines("BTCUSDT", klines)

        df = store.load("BTCUSDT")
        expected = {
            "pair",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "funding_rate",
            "open_interest",
            "trades",
        }
        assert set(df.columns) == expected
        store.close()

    def test_spot_backward_compat_alias(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbSpotStore

        store = DuckDbSpotStore(db_path=tmp_path / "s.duckdb")
        assert store.data_type == "spot"
        store.close()

    def test_schema_migration_spot_to_futures(self, tmp_path: Path):
        """Opening a spot DB as futures adds missing columns with NULL."""
        from signalflow.data.raw_store import DuckDbRawStore

        db = tmp_path / "upgrade.duckdb"

        # Create as spot and insert data
        spot = DuckDbRawStore(db_path=db, data_type="spot")
        klines = [self._make_kline(i) for i in range(5)]
        spot.insert_klines("BTCUSDT", klines)
        spot.close()

        # Re-open as futures
        futures = DuckDbRawStore(db_path=db, data_type="futures")
        df = futures.load("BTCUSDT")
        assert "open_interest" in df.columns
        assert len(df) == 5
        # Old rows should have NULL for open_interest
        assert df["open_interest"][0] is None
        futures.close()

    def test_futures_load_many(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "f.duckdb", data_type="futures")
        for pair in ["BTCUSDT", "ETHUSDT"]:
            klines = [self._make_kline(i, open_interest=50000.0) for i in range(10)]
            store.insert_klines(pair, klines)

        df = store.load_many(["BTCUSDT", "ETHUSDT"])
        assert len(df) == 20
        assert "open_interest" in df.columns
        assert set(df["pair"].unique().to_list()) == {"BTCUSDT", "ETHUSDT"}
        store.close()

    def test_futures_empty_load_many(self, tmp_path: Path):
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "f.duckdb", data_type="futures")
        df = store.load_many([])
        assert len(df) == 0
        assert "open_interest" in df.columns
        store.close()

    def test_futures_bulk_insert(self, tmp_path: Path):
        """Test bulk (>10 rows) Arrow-based insert for futures."""
        from signalflow.data.raw_store import DuckDbRawStore

        store = DuckDbRawStore(db_path=tmp_path / "f.duckdb", data_type="futures")
        klines = [self._make_kline(i, open_interest=50000.0 + i) for i in range(100)]
        store.insert_klines("BTCUSDT", klines)

        df = store.load("BTCUSDT")
        assert len(df) == 100
        assert df["open_interest"].is_not_null().all()
        store.close()
