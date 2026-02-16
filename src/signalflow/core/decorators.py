from typing import Any

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry


def sf_component(*, name: str, override: bool = True):
    """Register class as SignalFlow component.

    Decorator that registers a class in the global component registry,
    making it discoverable by name for dynamic instantiation.

    The decorated class must have a `component_type` class attribute
    of type `SfComponentType` to indicate what kind of component it is
    (e.g., DETECTOR, FEATURE, LABELER, STRATEGY_ENTRY_RULE, STRATEGY_EXIT_RULE).

    Args:
        name (str): Registry name for the component (case-insensitive).
        override (bool): Allow overriding existing registration. Default: True.

    Returns:
        Callable: Decorator function that registers and returns the class unchanged.

    Raises:
        ValueError: If class doesn't define component_type attribute.
        ValueError: If name already registered and override=False.

    Example:
        ```python
        from signalflow.core import sf_component
        from signalflow.core.enums import SfComponentType
        from signalflow.detector import SignalDetector

        @sf_component(name="my_detector")
        class MyDetector(SignalDetector):
            component_type = SfComponentType.DETECTOR

            def detect(self, df):
                # Detection logic
                return signals

        # Later, instantiate by name
        from signalflow.core.registry import default_registry

        detector_cls = default_registry.get(
            SfComponentType.DETECTOR,
            "my_detector"
        )
        detector = detector_cls(params={"window": 20})

        # Override existing registration
        @sf_component(name="my_detector", override=True)
        class ImprovedDetector(SignalDetector):
            component_type = SfComponentType.DETECTOR
            # ... improved implementation
        ```

    Example:
        ```python
        # Register multiple component types

        @sf_component(name="sma_cross")
        class SmaCrossDetector(SignalDetector):
            component_type = SfComponentType.DETECTOR
            # ...

        @sf_component(name="rsi")
        class RsiExtractor(FeatureExtractor):
            component_type = SfComponentType.FEATURE
            # ...

        @sf_component(name="fixed_size")
        class FixedSizeEntry(SignalEntryRule):
            component_type = SfComponentType.STRATEGY_ENTRY_RULE
            # ...

        @sf_component(name="take_profit")
        class TakeProfitExit(ExitRule):
            component_type = SfComponentType.STRATEGY_EXIT_RULE
            # ...
        ```

    Note:
        Component names are case-insensitive for lookup.
        The class itself is not modified - only registered.
        Use override=True carefully to avoid accidental overrides.
    """

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
