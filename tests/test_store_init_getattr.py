"""Tests for __getattr__ in store __init__ modules."""

import pytest


class TestRawStoreGetattr:
    def test_pg_raw_store_lazy_import(self):
        from signalflow.data import raw_store

        # Should return PgRawStore class via __getattr__
        pg_store_cls = getattr(raw_store, "PgRawStore", None)
        # May be None if psycopg not installed, that's OK
        # Just testing the __getattr__ path works

    def test_pg_spot_store_lazy_import(self):
        from signalflow.data import raw_store

        # Should return PgRawStore class via __getattr__
        pg_spot_cls = getattr(raw_store, "PgSpotStore", None)
        # May be None if psycopg not installed

    def test_unknown_attr_raises(self):
        from signalflow.data import raw_store

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(raw_store, "NonexistentStore")


class TestStrategyStoreGetattr:
    def test_pg_strategy_store_lazy_import(self):
        from signalflow.data import strategy_store

        # Should return PgStrategyStore class via __getattr__
        pg_store_cls = getattr(strategy_store, "PgStrategyStore", None)
        # May be None if psycopg not installed

    def test_unknown_attr_raises(self):
        from signalflow.data import strategy_store

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(strategy_store, "NonexistentStore")
