"""Component registration decorators.

This module provides semantic decorators for registering SignalFlow components.
Each decorator corresponds to a specific component type, making the code more
readable and providing better IDE support.

Example:
    ```python
    import signalflow as sf

    @sf.detector("sma_cross")
    class SmaCrossDetector(SignalDetector):
        def detect(self, df):
            ...

    @sf.entry("signal")
    class SignalEntry(EntryRule):
        def should_enter(self, signal):
            ...

    @sf.exit("tp_sl")
    class TpSlExit(ExitRule):
        def should_exit(self, position):
            ...
    ```

Migration from @sf_component:
    ```python
    # Before (deprecated)
    @sf_component(name="sma_cross")
    class SmaCross(SignalDetector):
        component_type = SfComponentType.DETECTOR

    # After
    @sf.detector("sma_cross")
    class SmaCross(SignalDetector):
        ...
    ```
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=type)


# =============================================================================
# Factory for semantic decorators
# =============================================================================


def _make_component_decorator(component_type: SfComponentType):
    """Factory for creating component-specific decorators.

    Args:
        component_type: The SfComponentType this decorator registers.

    Returns:
        A decorator factory function.
    """

    def decorator_factory(name: str, *, override: bool = True):
        """Register a class as a SignalFlow component.

        Args:
            name: Registry name for the component (case-insensitive).
            override: Allow overriding existing registration. Default: True.

        Returns:
            Decorator function that registers and returns the class unchanged.
        """

        def decorator(cls: T) -> T:
            # Set component_type on class for backward compatibility
            if not hasattr(cls, "component_type"):
                cls.component_type = component_type  # type: ignore[attr-defined]

            default_registry.register(
                component_type,
                name=name,
                cls=cls,
                override=override,
            )
            return cls

        return decorator

    # Add metadata for introspection
    decorator_factory._component_type = component_type  # type: ignore[attr-defined]

    return decorator_factory


# =============================================================================
# Semantic decorators (new API)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Signal Pipeline
# ─────────────────────────────────────────────────────────────────────────────

feature = _make_component_decorator(SfComponentType.FEATURE)
"""Register a feature extractor.

Example:
    ```python
    @sf.feature("rsi")
    class RsiFeature(FeatureExtractor):
        def compute(self, df):
            ...
    ```
"""

detector = _make_component_decorator(SfComponentType.DETECTOR)
"""Register a signal detector.

Example:
    ```python
    @sf.detector("sma_cross")
    class SmaCrossDetector(SignalDetector):
        def detect(self, df):
            ...
    ```
"""

validator = _make_component_decorator(SfComponentType.VALIDATOR)
"""Register a signal validator.

Example:
    ```python
    @sf.validator("lightgbm")
    class LightGBMValidator(SignalValidator):
        def validate(self, signals):
            ...
    ```
"""

labeler = _make_component_decorator(SfComponentType.LABELER)
"""Register a labeler.

Example:
    ```python
    @sf.labeler("fixed_horizon")
    class FixedHorizonLabeler(Labeler):
        def label(self, signals, prices):
            ...
    ```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

entry = _make_component_decorator(SfComponentType.STRATEGY_ENTRY_RULE)
"""Register an entry rule.

Example:
    ```python
    @sf.entry("signal")
    class SignalEntry(EntryRule):
        def should_enter(self, signal):
            ...
    ```
"""

exit = _make_component_decorator(SfComponentType.STRATEGY_EXIT_RULE)
"""Register an exit rule.

Example:
    ```python
    @sf.exit("tp_sl")
    class TpSlExit(ExitRule):
        def should_exit(self, position):
            ...
    ```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

signal_metric = _make_component_decorator(SfComponentType.SIGNAL_METRIC)
"""Register a signal quality metric.

Example:
    ```python
    @sf.signal_metric("hit_rate")
    class HitRateMetric(SignalMetric):
        def compute(self, signals, prices):
            ...
    ```
"""

strategy_metric = _make_component_decorator(SfComponentType.STRATEGY_METRIC)
"""Register a strategy performance metric.

Example:
    ```python
    @sf.strategy_metric("sharpe")
    class SharpeMetric(StrategyMetric):
        def compute(self, trades):
            ...
    ```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Hooks & Alerts
# ─────────────────────────────────────────────────────────────────────────────

alert = _make_component_decorator(SfComponentType.STRATEGY_ALERT)
"""Register a strategy alert.

Example:
    ```python
    @sf.alert("max_drawdown")
    class MaxDrawdownAlert(Alert):
        def check(self, state):
            ...
    ```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data & Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

data_source = _make_component_decorator(SfComponentType.RAW_DATA_SOURCE)
"""Register a data source (e.g., Binance API).

Example:
    ```python
    @sf.data_source("binance")
    class BinanceSource(DataSource):
        ...
    ```
"""

data_store = _make_component_decorator(SfComponentType.RAW_DATA_STORE)
"""Register a data store (e.g., DuckDB).

Example:
    ```python
    @sf.data_store("duckdb")
    class DuckDBStore(DataStore):
        ...
    ```
"""

strategy_store = _make_component_decorator(SfComponentType.STRATEGY_STORE)
"""Register a strategy state store.

Example:
    ```python
    @sf.strategy_store("redis")
    class RedisStrategyStore(StrategyStore):
        ...
    ```
"""

executor = _make_component_decorator(SfComponentType.STRATEGY_EXECUTOR)
"""Register a strategy executor.

Example:
    ```python
    @sf.executor("binance_spot")
    class BinanceSpotExecutor(Executor):
        ...
    ```
"""

risk = _make_component_decorator(SfComponentType.STRATEGY_RISK)
"""Register a risk limit.

Example:
    ```python
    @sf.risk("max_positions")
    class MaxPositionsLimit(RiskLimit):
        ...
    ```
"""


# =============================================================================
# Legacy decorator (deprecated)
# =============================================================================


def sf_component(*, name: str, override: bool = True):
    """Register class as SignalFlow component.

    .. deprecated:: 0.6.0
        Use semantic decorators instead:
        - @sf.detector("name") for detectors
        - @sf.feature("name") for features
        - @sf.entry("name") for entry rules
        - @sf.exit("name") for exit rules
        - etc.

    Decorator that registers a class in the global component registry,
    making it discoverable by name for dynamic instantiation.

    The decorated class must have a `component_type` class attribute
    of type `SfComponentType` to indicate what kind of component it is.

    Args:
        name: Registry name for the component (case-insensitive).
        override: Allow overriding existing registration. Default: True.

    Returns:
        Decorator function that registers and returns the class unchanged.

    Raises:
        ValueError: If class doesn't define component_type attribute.
    """
    warnings.warn(
        "@sf_component is deprecated. Use semantic decorators instead:\n"
        "  @sf.detector('name'), @sf.feature('name'), @sf.entry('name'), etc.\n"
        "See https://signalflow-trading.com/migration for details.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(cls: type[Any]) -> type[Any]:
        component_type = getattr(cls, "component_type", None)
        if not isinstance(component_type, SfComponentType):
            raise ValueError(f"{cls.__name__} must define class attribute 'component_type: SfComponentType'")

        default_registry.register(
            component_type,
            name=name,
            cls=cls,
            override=override,
        )
        return cls

    return decorator


# =============================================================================
# Generic register (type inferred from class)
# =============================================================================


def register(name: str, *, override: bool = True):
    """Register component with auto-detected type.

    Infers component_type from the class's component_type attribute
    or from its base class. Use this when you want a single decorator
    that works with any component type.

    Args:
        name: Registry name for the component.
        override: Allow overriding existing registration. Default: True.

    Returns:
        Decorator function.

    Example:
        ```python
        @sf.register("my_detector")
        class MyDetector(SignalDetector):
            component_type = SfComponentType.DETECTOR
            ...
        ```

    Note:
        Prefer semantic decorators (@sf.detector, @sf.entry, etc.) for clarity.
        Use @sf.register only when you need generic registration.
    """

    def decorator(cls: T) -> T:
        component_type = getattr(cls, "component_type", None)

        if component_type is None:
            # Try to infer from base class
            component_type = _infer_component_type(cls)

        if component_type is None:
            raise ValueError(
                f"Cannot infer component_type for {cls.__name__}. "
                "Either define 'component_type' class attribute or use "
                "a semantic decorator (@sf.detector, @sf.entry, etc.)."
            )

        default_registry.register(
            component_type,
            name=name,
            cls=cls,
            override=override,
        )
        return cls

    return decorator


def _infer_component_type(cls: type) -> SfComponentType | None:
    """Attempt to infer component type from base classes.

    Checks MRO for known base classes and returns corresponding type.
    """
    # Import here to avoid circular imports
    from signalflow.core.enums import SfComponentType

    # Map of base class names to component types
    base_to_type: dict[str, SfComponentType] = {
        "SignalDetector": SfComponentType.DETECTOR,
        "FeatureExtractor": SfComponentType.FEATURE,
        "Feature": SfComponentType.FEATURE,
        "SignalValidator": SfComponentType.VALIDATOR,
        "Labeler": SfComponentType.LABELER,
        "EntryRule": SfComponentType.STRATEGY_ENTRY_RULE,
        "SignalEntryRule": SfComponentType.STRATEGY_ENTRY_RULE,
        "ExitRule": SfComponentType.STRATEGY_EXIT_RULE,
        "SignalMetric": SfComponentType.SIGNAL_METRIC,
        "StrategyMetric": SfComponentType.STRATEGY_METRIC,
        "Alert": SfComponentType.STRATEGY_ALERT,
        "DataSource": SfComponentType.RAW_DATA_SOURCE,
        "DataStore": SfComponentType.RAW_DATA_STORE,
        "StrategyStore": SfComponentType.STRATEGY_STORE,
        "Executor": SfComponentType.STRATEGY_EXECUTOR,
        "RiskLimit": SfComponentType.STRATEGY_RISK,
    }

    for base in cls.__mro__:
        if base.__name__ in base_to_type:
            return base_to_type[base.__name__]

    return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "alert",
    "data_source",
    "data_store",
    "detector",
    "entry",
    "executor",
    "exit",
    # Semantic decorators (new API)
    "feature",
    "labeler",
    # Generic
    "register",
    "risk",
    # Legacy (deprecated)
    "sf_component",
    "signal_metric",
    "strategy_metric",
    "strategy_store",
    "validator",
]
