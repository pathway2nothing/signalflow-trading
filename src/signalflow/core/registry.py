from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type
from loguru import logger
from .enums import SfComponentType


@dataclass
class SignalFlowRegistry:
    """Component registry for dynamic component discovery and instantiation.

    Provides centralized registration and lookup for SignalFlow components.
    Components are organized by type (DETECTOR, EXTRACTOR, etc.) and 
    accessed by case-insensitive names.

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

    Example:
        ```python
        from signalflow.core.registry import SignalFlowRegistry
        from signalflow.core.enums import SfComponentType

        # Create registry
        registry = SignalFlowRegistry()

        # Register component
        registry.register(
            SfComponentType.DETECTOR,
            name="sma_cross",
            cls=SmaCrossDetector
        )

        # Get component class
        detector_cls = registry.get(SfComponentType.DETECTOR, "sma_cross")

        # Instantiate component
        detector = registry.create(
            SfComponentType.DETECTOR,
            "sma_cross",
            fast_window=10,
            slow_window=20
        )

        # List available components
        detectors = registry.list(SfComponentType.DETECTOR)
        print(f"Available detectors: {detectors}")

        # Full snapshot
        snapshot = registry.snapshot()
        print(snapshot)
        ```

    Note:
        Component names are stored and looked up in lowercase.
        Use default_registry singleton for application-wide registration.

    See Also:
        sf_component: Decorator for automatic component registration.
    """
    #TODO: Registry autodiscover

    _items: Dict[SfComponentType, Dict[str, Type[Any]]] = field(default_factory=dict)

    def _ensure(self, component_type: SfComponentType) -> None:
        """Ensure component type exists in registry.

        Initializes empty dict for component_type if not present.

        Args:
            component_type (SfComponentType): Component type to ensure.
        """
        self._items.setdefault(component_type, {})

    def register(self, component_type: SfComponentType, name: str, cls: Type[Any], *, override: bool = False) -> None:
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

    def get(self, component_type: SfComponentType, name: str) -> Type[Any]:
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
        self._ensure(component_type)
        key = name.lower()
        try:
            return self._items[component_type][key]
        except KeyError as e:
            available = ", ".join(sorted(self._items[component_type]))
            raise KeyError(
                f"Component not found: {component_type.value}:{key}. Available: [{available}]"
            ) from e

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
        self._ensure(component_type)
        return sorted(self._items[component_type])

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

def get_component(type: SfComponentType, name: str  ) -> Type[Any]:
    """Get a registered component by type and name."""
    return default_registry.get(type, name) 