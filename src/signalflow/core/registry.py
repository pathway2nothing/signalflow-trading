from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from loguru import logger

from .enums import SfComponentType

_BUILTIN_RAW_DATA_TYPES: dict[str, set[str]] = {
    "spot": {"pair", "timestamp", "open", "high", "low", "close", "volume"},
    "futures": {"pair", "timestamp", "open", "high", "low", "close", "volume", "open_interest"},
    "perpetual": {"pair", "timestamp", "open", "high", "low", "close", "volume", "funding_rate", "open_interest"},
}


@dataclass
class SignalFlowRegistry:
    """Component registry for dynamic component discovery and instantiation.

    Provides centralized registration and lookup for SignalFlow components.
    Components are organized by type (DETECTOR, EXTRACTOR, etc.) and
    accessed by case-insensitive names.

    Also manages extensible raw data type definitions - each data type maps
    to a set of required columns. Built-in types (SPOT, FUTURES, PERPETUAL)
    are pre-registered; users can add custom types via ``register_raw_data_type()``.

    Registry structure:
        component_type -> name -> class

    Supported component types:
        - DETECTOR: Signal detection classes
        - EXTRACTOR: Feature extraction classes
        - LABELER: Signal labeling classes
        - ENTRY_RULE: Position entry rules
        - EXIT_RULE: Position exit rules
        - METRIC: Strategy metrics
        - EXECUTOR: Order execution engines

    Attributes:
        _items (dict[SfComponentType, dict[str, Type[Any]]]):
            Internal storage mapping component types to name-class pairs.
        _raw_data_types (dict[str, set[str]]):
            Mapping of raw data type names to their required column sets.

    Example:
        ```python
        from signalflow.core.registry import SignalFlowRegistry, default_registry

        # Register custom raw data type
        default_registry.register_raw_data_type(
            name="lob",
            columns=["pair", "timestamp", "bid", "ask", "bid_size", "ask_size"],
        )

        # Get columns for any type
        cols = default_registry.get_raw_data_columns("spot")
        custom_cols = default_registry.get_raw_data_columns("lob")

        # List all registered raw data types
        print(default_registry.list_raw_data_types())
        ```

    Note:
        Component names are stored and looked up in lowercase.
        Use default_registry singleton for application-wide registration.

    See Also:
        sf_component: Decorator for automatic component registration.
    """

    _items: dict[SfComponentType, dict[str, type[Any]]] = field(default_factory=dict)
    _raw_data_types: dict[str, set[str]] = field(
        default_factory=lambda: {k: v.copy() for k, v in _BUILTIN_RAW_DATA_TYPES.items()}
    )
    _discovered: bool = field(default=False, repr=False)

    def _ensure(self, component_type: SfComponentType) -> None:
        """Ensure component type exists in registry.

        Initializes empty dict for component_type if not present.

        Args:
            component_type (SfComponentType): Component type to ensure.
        """
        self._items.setdefault(component_type, {})

    # ── Autodiscovery ─────────────────────────────────────────────────

    def _discover_if_needed(self) -> None:
        """Trigger autodiscovery on first read access (lazy loading)."""
        if not self._discovered:
            self.autodiscover()

    def autodiscover(self) -> None:
        """Scan ``signalflow.*`` packages and entry-points for components.

        Walks all sub-modules of the ``signalflow`` package using
        :func:`pkgutil.walk_packages` and imports them.  Because
        :func:`sf_component` registers classes at import time, importing
        a module is sufficient to populate the registry.

        External packages can expose components via the
        ``signalflow.components`` entry-point group.  Each entry-point
        should reference a module (not a callable); importing it triggers
        registration through the ``@sf_component`` decorator.

        This method is idempotent - subsequent calls are no-ops once
        ``_discovered`` is ``True``.

        Example:
            ```python
            from signalflow.core.registry import default_registry

            # Explicit discovery (normally automatic on first get/list)
            default_registry.autodiscover()

            # All @sf_component-decorated classes are now registered
            print(default_registry.snapshot())
            ```
        """
        if self._discovered:
            return
        self._discovered = True

        self._discover_internal_packages()
        self._discover_entry_points()

    def _discover_internal_packages(self) -> None:
        """Import all ``signalflow.*`` sub-modules to trigger registration."""
        try:
            import signalflow as _sf_root
        except ImportError:  # pragma: no cover
            logger.warning("signalflow package not importable - skipping internal autodiscovery")
            return

        pkg_path = getattr(_sf_root, "__path__", None)
        if pkg_path is None:  # pragma: no cover
            return

        for _importer, modname, _ispkg in pkgutil.walk_packages(pkg_path, prefix="signalflow."):
            if modname in sys.modules:
                continue
            try:
                importlib.import_module(modname)
            except Exception:
                logger.debug(f"autodiscover: failed to import {modname}")

    def _discover_entry_points(self) -> None:
        """Load external plugins registered under ``signalflow.components``."""
        eps = entry_points()

        # Python 3.12+: entry_points() returns SelectableGroups
        if hasattr(eps, "select"):
            group = eps.select(group="signalflow.components")
        else:  # pragma: no cover
            group = eps.get("signalflow.components", [])

        for ep in group:
            try:
                ep.load()
            except Exception:
                logger.warning(f"autodiscover: failed to load entry-point {ep.name!r}")

    def register(self, component_type: SfComponentType, name: str, cls: type[Any], *, override: bool = False) -> None:
        """Register a class under (component_type, name).

        Stores class in registry for later lookup and instantiation.
        Names are normalized to lowercase for case-insensitive lookup.

        Args:
            component_type (SfComponentType): Type of component (DETECTOR, EXTRACTOR, etc.).
            name (str): Registry name (case-insensitive, will be lowercased).
            cls (Type[Any]): Class to register.
            override (bool): Allow overriding existing registration. Default: False.

        Raises:
            ValueError: If name is empty or already registered (when override=False).

        Example:
            ```python
            # Register new component
            registry.register(
                SfComponentType.DETECTOR,
                name="my_detector",
                cls=MyDetector
            )

            # Override existing component
            registry.register(
                SfComponentType.DETECTOR,
                name="my_detector",
                cls=ImprovedDetector,
                override=True  # Logs warning
            )

            # Register multiple types
            registry.register(SfComponentType.EXTRACTOR, "rsi", RsiExtractor)
            registry.register(SfComponentType.LABELER, "fixed", FixedHorizonLabeler)
            ```
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        key = name.strip().lower()
        self._ensure(component_type)

        if key in self._items[component_type] and not override:
            raise ValueError(f"{component_type.value}:{key} already registered")

        if key in self._items[component_type] and override:
            logger.warning(f"Overriding {component_type.value}:{key} with {cls.__name__}")

        self._items[component_type][key] = cls

    def get(self, component_type: SfComponentType, name: str) -> type[Any]:
        """Get a registered class by key.

        Lookup is case-insensitive. Raises helpful error with available
        components if key not found.

        Args:
            component_type (SfComponentType): Type of component to lookup.
            name (str): Component name (case-insensitive).

        Returns:
            Type[Any]: Registered class.

        Raises:
            KeyError: If component not found. Error message includes available components.

        Example:
            ```python
            # Get component class
            detector_cls = registry.get(SfComponentType.DETECTOR, "sma_cross")

            # Case-insensitive
            detector_cls = registry.get(SfComponentType.DETECTOR, "SMA_Cross")

            # Instantiate manually
            detector = detector_cls(fast_window=10, slow_window=20)

            # Handle missing component
            try:
                cls = registry.get(SfComponentType.DETECTOR, "unknown")
            except KeyError as e:
                print(f"Component not found: {e}")
                # Shows: "Component not found: DETECTOR:unknown. Available: [sma_cross, ...]"
            ```
        """
        self._discover_if_needed()
        self._ensure(component_type)
        key = name.lower()
        try:
            return self._items[component_type][key]
        except KeyError as e:
            available = ", ".join(sorted(self._items[component_type]))
            raise KeyError(f"Component not found: {component_type.value}:{key}. Available: [{available}]") from e

    def create(self, component_type: SfComponentType, name: str, **kwargs: Any) -> Any:
        """Instantiate a component by registry key.

        Convenient method that combines get() and instantiation.

        Args:
            component_type (SfComponentType): Type of component to create.
            name (str): Component name (case-insensitive).
            **kwargs: Arguments to pass to component constructor.

        Returns:
            Any: Instantiated component.

        Raises:
            KeyError: If component not found.
            TypeError: If kwargs don't match component constructor.

        Example:
            ```python
            # Create detector with params
            detector = registry.create(
                SfComponentType.DETECTOR,
                "sma_cross",
                fast_window=10,
                slow_window=20
            )

            # Create extractor
            extractor = registry.create(
                SfComponentType.EXTRACTOR,
                "rsi",
                window=14
            )

            # Create with config dict
            config = {"window": 20, "threshold": 0.7}
            labeler = registry.create(
                SfComponentType.LABELER,
                "fixed",
                **config
            )
            ```
        """
        cls = self.get(component_type, name)
        return cls(**kwargs)

    def list(self, component_type: SfComponentType) -> list[str]:
        """List registered components for a type.

        Returns sorted list of component names for given type.

        Args:
            component_type (SfComponentType): Type of components to list.

        Returns:
            list[str]: Sorted list of registered component names.

        Example:
            ```python
            # List all detectors
            detectors = registry.list(SfComponentType.DETECTOR)
            print(f"Available detectors: {detectors}")
            # Output: ['ema_cross', 'macd', 'rsi_threshold', 'sma_cross']

            # Check if component exists
            if "sma_cross" in registry.list(SfComponentType.DETECTOR):
                detector = registry.create(SfComponentType.DETECTOR, "sma_cross")

            # List all component types
            from signalflow.core.enums import SfComponentType
            for component_type in SfComponentType:
                components = registry.list(component_type)
                print(f"{component_type.value}: {components}")
            ```
        """
        self._discover_if_needed()
        self._ensure(component_type)
        return sorted(self._items[component_type])

    # ── Raw data type registry ─────────────────────────────────────────

    def register_raw_data_type(
        self,
        name: str,
        columns: list[str] | set[str],
        *,
        override: bool = False,
    ) -> None:
        """Register a custom raw data type with its required columns.

        Args:
            name: Data type identifier (case-insensitive, stored lowercase).
            columns: Required column names for this data type.
            override: Allow overriding an existing registration.

        Raises:
            ValueError: If name is empty, columns are empty, or name already
                registered (when override=False).

        Example:
            ```python
            default_registry.register_raw_data_type(
                name="lob",
                columns=["pair", "timestamp", "bid", "ask", "bid_size", "ask_size"],
            )
            ```
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")

        cols = set(columns)
        if not cols:
            raise ValueError("columns must be a non-empty collection")

        key = name.strip().lower()

        if key in self._raw_data_types and not override:
            raise ValueError(f"Raw data type '{key}' already registered")

        if key in self._raw_data_types and override:
            logger.warning(f"Overriding raw data type '{key}'")

        self._raw_data_types[key] = cols

    def get_raw_data_columns(self, name: str) -> set[str]:
        """Get required columns for a raw data type.

        Args:
            name: Data type identifier (case-insensitive). Accepts both
                ``RawDataType`` enum members and plain strings.

        Returns:
            Copy of the column set for the requested type.

        Raises:
            KeyError: If data type is not registered.

        Example:
            ```python
            cols = default_registry.get_raw_data_columns("spot")
            # {'pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}

            cols = default_registry.get_raw_data_columns("lob")
            # {'pair', 'timestamp', 'bid', 'ask', ...}
            ```
        """
        raw = getattr(name, "value", name)
        key = str(raw).strip().lower()
        try:
            return self._raw_data_types[key].copy()
        except KeyError:
            available = ", ".join(sorted(self._raw_data_types))
            raise KeyError(f"Raw data type '{key}' not registered. Available: [{available}]") from None

    def list_raw_data_types(self) -> list[str]:
        """List all registered raw data type names.

        Returns:
            Sorted list of registered data type names.
        """
        return sorted(self._raw_data_types)

    def snapshot(self) -> dict[str, list[str]]:
        """Snapshot of registry for debugging.

        Returns complete registry state organized by component type.

        Returns:
            dict[str, list[str]]: Dictionary mapping component type names
                to sorted lists of registered component names.

        Example:
            ```python
            # Get full registry snapshot
            snapshot = registry.snapshot()
            print(snapshot)
            # Output:
            # {
            #     'DETECTOR': ['ema_cross', 'sma_cross'],
            #     'EXTRACTOR': ['rsi', 'sma'],
            #     'LABELER': ['fixed', 'triple_barrier'],
            #     'ENTRY_RULE': ['fixed_size'],
            #     'EXIT_RULE': ['take_profit', 'time_based']
            # }

            # Use for debugging
            import json
            print(json.dumps(registry.snapshot(), indent=2))

            # Check registration status
            snapshot = registry.snapshot()
            if 'DETECTOR' in snapshot and 'sma_cross' in snapshot['DETECTOR']:
                print("SMA detector is registered")
            ```
        """
        self._discover_if_needed()
        return {t.value: sorted(v.keys()) for t, v in self._items.items()}


default_registry = SignalFlowRegistry()
"""Global default registry instance.

Use this singleton for application-wide component registration.

Example:
    ```python
    from signalflow.core.registry import default_registry
    from signalflow.core.enums import SfComponentType

    # Register to default registry
    default_registry.register(
        SfComponentType.DETECTOR,
        "my_detector",
        MyDetector
    )

    # Access from anywhere
    detector = default_registry.create(
        SfComponentType.DETECTOR,
        "my_detector"
    )
    ```
"""


def get_component(type: SfComponentType, name: str) -> type[Any]:
    """Get a registered component by type and name."""
    return default_registry.get(type, name)
