"""Tests for RawDataLazy."""

import warnings
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from signalflow.core.containers.raw_data_lazy import RawDataLazy, LazyDataTypeAccessor
from signalflow.data.raw_store import DuckDbRawStore


def _make_perpetual_data(n=10, oi_multiplier=1.0):
    """Helper to create perpetual klines."""
    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, i),
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 1000.0,
            "open_interest": 50000.0 * oi_multiplier,
            "funding_rate": 0.0001,
        }
        for i in range(n)
    ]


@pytest.fixture
def stores(tmp_path: Path):
    """Create test stores with data."""
    binance = DuckDbRawStore(db_path=tmp_path / "binance.duckdb", data_type="perpetual")
    okx = DuckDbRawStore(db_path=tmp_path / "okx.duckdb", data_type="perpetual")

    binance.insert_klines("BTCUSDT", _make_perpetual_data(10, 1.0))
    okx.insert_klines("BTCUSDT", _make_perpetual_data(10, 0.8))

    yield {"binance": binance, "okx": okx}

    binance.close()
    okx.close()


class TestRawDataLazyCreation:
    def test_from_stores(self, stores: dict, tmp_path: Path):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        assert raw.pairs == ["BTCUSDT"]
        assert raw.default_source == "binance"
        assert raw.cache_mode == "memory"

    def test_default_source_explicit(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            default_source="okx",
        )

        assert raw.default_source == "okx"


class TestLazyLoading:
    def test_not_loaded_until_accessed(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="memory",
        )

        # Nothing loaded yet
        assert raw.is_loaded == {"perpetual": {"binance": False, "okx": False}}

        # Access binance
        _ = raw.perpetual.binance

        # Only binance loaded
        assert raw.is_loaded == {"perpetual": {"binance": True, "okx": False}}

        # Access okx
        _ = raw.perpetual.okx

        # Both loaded
        assert raw.is_loaded == {"perpetual": {"binance": True, "okx": True}}

    def test_memory_cache_reused(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="memory",
        )

        # First access loads from store
        df1 = raw.perpetual.binance

        # Second access returns same object (from cache)
        df2 = raw.perpetual.binance

        assert df1 is df2  # Same object reference

    def test_no_cache_mode(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="none",
        )

        # Access twice
        df1 = raw.perpetual.binance
        df2 = raw.perpetual.binance

        # Different objects (loaded fresh each time)
        assert df1 is not df2

        # is_loaded always False
        assert raw.is_loaded == {"perpetual": {"binance": False, "okx": False}}

    def test_disk_cache_mode(self, stores: dict, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="disk",
            cache_dir=cache_dir,
        )

        # Load data
        df1 = raw.perpetual.binance

        # Check parquet file created
        assert any(cache_dir.glob("*.parquet"))

        # is_loaded shows True
        assert raw.is_loaded["perpetual"]["binance"] is True

        # Second load reads from parquet
        df2 = raw.perpetual.binance
        assert len(df1) == len(df2)


class TestLazyAccessorInterface:
    def test_sources_property(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        # sources doesn't load data
        assert raw.perpetual.sources == ["binance", "okx"]
        assert raw.is_loaded == {"perpetual": {"binance": False, "okx": False}}

    def test_accessor_getattr(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        df = raw.perpetual.binance
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert df["open_interest"][0] == 50000.0

    def test_accessor_to_polars_warns(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = raw.perpetual.to_polars()
            assert len(w) == 1
            assert "default source" in str(w[0].message)
            assert isinstance(df, pl.DataFrame)

    def test_accessor_iter(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        items = list(raw.perpetual)
        assert len(items) == 2
        assert items[0][0] == "binance"
        assert isinstance(items[0][1], pl.DataFrame)

    def test_accessor_contains(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        assert "binance" in raw.perpetual
        assert "unknown" not in raw.perpetual

    def test_accessor_len(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        assert len(raw.perpetual) == 2

    def test_accessor_missing_source_raises(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        with pytest.raises(AttributeError, match="No source 'unknown'"):
            _ = raw.perpetual.unknown


class TestDictStyleAccess:
    def test_getitem_tuple(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        df = raw["perpetual", "okx"]
        assert isinstance(df, pl.DataFrame)
        assert df["open_interest"][0] == 40000.0  # 50000 * 0.8

    def test_get_with_source(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        df = raw.get("perpetual", source="okx")
        assert isinstance(df, pl.DataFrame)

    def test_contains(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        assert "perpetual" in raw
        assert "spot" not in raw

    def test_keys(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        assert list(raw.keys()) == ["perpetual"]

    def test_sources_method(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        assert raw.sources("perpetual") == ["binance", "okx"]


class TestCacheManagement:
    def test_preload(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="memory",
        )

        assert raw.is_loaded == {"perpetual": {"binance": False, "okx": False}}

        raw.preload()

        assert raw.is_loaded == {"perpetual": {"binance": True, "okx": True}}

    def test_preload_selective(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="memory",
        )

        raw.preload(sources=["binance"])

        assert raw.is_loaded == {"perpetual": {"binance": True, "okx": False}}

    def test_clear_cache_memory(self, stores: dict):
        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="memory",
        )

        raw.preload()
        assert raw.is_loaded == {"perpetual": {"binance": True, "okx": True}}

        raw.clear_cache()
        assert raw.is_loaded == {"perpetual": {"binance": False, "okx": False}}

    def test_clear_cache_disk(self, stores: dict, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        raw = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
            cache_mode="disk",
            cache_dir=cache_dir,
        )

        raw.preload()
        assert any(cache_dir.glob("*.parquet"))

        raw.clear_cache()
        assert not any(cache_dir.glob("*.parquet"))


class TestConversion:
    def test_to_raw_data(self, stores: dict):
        raw_lazy = RawDataLazy.from_stores(
            stores=stores,
            pairs=["BTCUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 0, 9),
        )

        from signalflow.core import RawData

        raw_eager = raw_lazy.to_raw_data()

        assert isinstance(raw_eager, RawData)
        assert "perpetual" in raw_eager.data
        assert "binance" in raw_eager.data["perpetual"]
        assert "okx" in raw_eager.data["perpetual"]
