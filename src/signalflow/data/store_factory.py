"""Backend-agnostic store factory."""

from __future__ import annotations

import dataclasses
from typing import Any

from signalflow.core import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.raw_store.base import RawDataStore
from signalflow.data.strategy_store.base import StrategyStore

_RAW_STORE_BACKENDS: dict[str, str] = {
    "duckdb": "duckdb",
    "sqlite": "sqlite",
    "postgres": "postgres",
    "memory": "memory",
}

_STRATEGY_STORE_BACKENDS: dict[str, str] = {
    "duckdb": "duckdb/strategy",
    "sqlite": "sqlite/strategy",
    "postgres": "postgres/strategy",
    "memory": "memory/strategy",
}


class StoreFactory:
    """Factory for creating storage backends.

    Supported backends:
        - ``"duckdb"`` (default) - local DuckDB file storage
        - ``"sqlite"`` - local SQLite file storage (zero extra deps)
        - ``"postgres"`` - PostgreSQL (requires ``pip install signalflow-trading[postgres]``)
        - ``"memory"`` - in-memory storage (no persistence, useful for tests/notebooks)
    """

    @staticmethod
    def create_raw_store(backend: str = "duckdb", data_type: str = "spot", **kwargs: Any) -> RawDataStore:
        """Create a RawDataStore for the given backend.

        Args:
            backend: One of ``"duckdb"``, ``"sqlite"``, ``"postgres"``, ``"memory"``.
            data_type: Raw data type (``"spot"``, ``"futures"``, ``"perpetual"``).
            **kwargs: Backend-specific constructor arguments.

        Raises:
            KeyError: If backend is not recognised.
            ImportError: If optional dependencies are missing.
        """
        key = backend.lower()
        if key not in _RAW_STORE_BACKENDS:
            available = ", ".join(sorted(_RAW_STORE_BACKENDS))
            raise KeyError(f"Unknown raw store backend: {key!r}. Available: {available}")
        registry_name = f"{_RAW_STORE_BACKENDS[key]}/{data_type}"
        cls = default_registry.get(SfComponentType.RAW_DATA_STORE, registry_name)
        # Only pass data_type if the store class accepts it.
        if dataclasses.is_dataclass(cls) and "data_type" in {f.name for f in dataclasses.fields(cls)}:
            kwargs["data_type"] = data_type
        return cls(**kwargs)

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
