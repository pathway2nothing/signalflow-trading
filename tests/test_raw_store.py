"""Tests for RawDataStore implementations (DuckDB, SQLite)."""

from datetime import datetime, timedelta

import polars as pl


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
