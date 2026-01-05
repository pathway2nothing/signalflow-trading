from typing import Any, Type
from signalflow.core.registry import default_registry
from signalflow.core.enums import SfComponentType


def sf_component(*, name: str, override: bool = False):
    """Register class as SignalFlow component.

    Args:
        name: Registry name (case-insensitive).
        override: Allow overriding existing registration.
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        component_type = getattr(cls, "component_type", None)
        if not isinstance(component_type, SfComponentType):
            raise ValueError(
                f"{cls.__name__} must define class attribute "
                f"'component_type: SfComponentType'"
            )

        default_registry.register(
            component_type,
            name=name,
            cls=cls,
            override=override,
        )
        return cls

    return decorator
