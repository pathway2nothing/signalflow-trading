"""Backend-agnostic store factory."""

from __future__ import annotations

from typing import Any

from signalflow.core import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.strategy_store.base import StrategyStore

_RAW_STORE_BACKENDS: dict[str, str] = {
    "duckdb": "duckdb/spot",
    "sqlite": "sqlite/spot",
    "postgres": "postgres/spot",
}

_STRATEGY_STORE_BACKENDS: dict[str, str] = {
    "duckdb": "duckdb/strategy",
    "sqlite": "sqlite/strategy",
    "postgres": "postgres/strategy",
}


class StoreFactory:
    """Factory for creating storage backends.

    Supported backends:
        - ``"duckdb"`` (default) — local DuckDB file storage
        - ``"sqlite"`` — local SQLite file storage (zero extra deps)
        - ``"postgres"`` — PostgreSQL (requires ``pip install signalflow-trading[postgres]``)
    """

    @staticmethod
    def create_raw_store(backend: str = "duckdb", **kwargs: Any) -> RawDataStore:
        """Create a RawDataStore for the given backend.

        Args:
            backend: One of ``"duckdb"``, ``"sqlite"``, ``"postgres"``.
            **kwargs: Backend-specific constructor arguments.

        Raises:
            KeyError: If backend is not recognised.
            ImportError: If optional dependencies are missing.
        """
        key = backend.lower()
        if key not in _RAW_STORE_BACKENDS:
            available = ", ".join(sorted(_RAW_STORE_BACKENDS))
            raise KeyError(f"Unknown raw store backend: {key!r}. Available: {available}")
        registry_name = _RAW_STORE_BACKENDS[key]
        return default_registry.create(SfComponentType.RAW_DATA_STORE, registry_name, **kwargs)

    @staticmethod
    def create_strategy_store(backend: str = "duckdb", **kwargs: Any) -> StrategyStore:
        """Create a StrategyStore for the given backend.

        Args:
            backend: One of ``"duckdb"``, ``"sqlite"``, ``"postgres"``.
            **kwargs: Backend-specific constructor arguments.

        Raises:
            KeyError: If backend is not recognised.
            ImportError: If optional dependencies are missing.
        """
        key = backend.lower()
        if key not in _STRATEGY_STORE_BACKENDS:
            available = ", ".join(sorted(_STRATEGY_STORE_BACKENDS))
            raise KeyError(f"Unknown strategy store backend: {key!r}. Available: {available}")
        registry_name = _STRATEGY_STORE_BACKENDS[key]
        return default_registry.create(SfComponentType.STRATEGY_STORE, registry_name, **kwargs)
