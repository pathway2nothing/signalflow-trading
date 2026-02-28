"""Tests for __getattr__ in store __init__ modules."""

import pytest


class TestRawStoreGetattr:
    def test_pg_raw_store_lazy_import(self):
        from signalflow.data import raw_store

        # __getattr__ returns the class if sf-data is installed,
        # raises ImportError if not
        try:
            cls = raw_store.PgRawStore
            assert cls is not None
        except ImportError:
            pass  # sf-data not installed, expected

    def test_pg_spot_store_lazy_import(self):
        from signalflow.data import raw_store

        try:
            cls = raw_store.PgSpotStore
            assert cls is not None
        except ImportError:
            pass  # sf-data not installed, expected

    def test_unknown_attr_raises(self):
        from signalflow.data import raw_store

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = raw_store.NonexistentStore


class TestStrategyStoreGetattr:
    def test_pg_strategy_store_lazy_import(self):
        from signalflow.data import strategy_store

        try:
            cls = strategy_store.PgStrategyStore
            assert cls is not None
        except ImportError:
            pass  # sf-data not installed, expected

    def test_unknown_attr_raises(self):
        from signalflow.data import strategy_store

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = strategy_store.NonexistentStore
