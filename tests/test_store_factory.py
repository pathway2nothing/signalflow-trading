"""Tests for StoreFactory."""

import pytest

from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.store_factory import StoreFactory
from signalflow.data.strategy_store.base import StrategyStore

_SF_DATA_AVAILABLE = True
try:
    from signalflow.data.raw_store import SqliteRawStore  # noqa: F401
except ImportError:
    _SF_DATA_AVAILABLE = False

needs_sf_data = pytest.mark.skipif(not _SF_DATA_AVAILABLE, reason="signalflow-data package not installed")


class TestStoreFactory:
    def test_create_duckdb_raw_store(self, tmp_path):
        store = StoreFactory.create_raw_store("duckdb", db_path=tmp_path / "test.duckdb")
        assert isinstance(store, RawDataStore)
        store.close()

    @needs_sf_data
    def test_create_sqlite_raw_store(self, tmp_path):
        store = StoreFactory.create_raw_store("sqlite", db_path=tmp_path / "test.sqlite")
        assert isinstance(store, RawDataStore)
        store.close()

    def test_create_duckdb_strategy_store(self, tmp_path):
        store = StoreFactory.create_strategy_store("duckdb", path=str(tmp_path / "test.duckdb"))
        assert isinstance(store, StrategyStore)
        store.close()

    @needs_sf_data
    def test_create_sqlite_strategy_store(self, tmp_path):
        store = StoreFactory.create_strategy_store("sqlite", path=str(tmp_path / "test.sqlite"))
        assert isinstance(store, StrategyStore)
        store.close()

    def test_unknown_raw_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown raw store backend"):
            StoreFactory.create_raw_store("redis")

    def test_unknown_strategy_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown strategy store backend"):
            StoreFactory.create_strategy_store("redis")

    @needs_sf_data
    def test_case_insensitive_backend(self, tmp_path):
        store = StoreFactory.create_raw_store("SQLite", db_path=tmp_path / "test.sqlite")
        assert isinstance(store, RawDataStore)
        store.close()
