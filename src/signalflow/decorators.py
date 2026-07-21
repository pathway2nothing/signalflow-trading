"""Semantic registration decorators."""

import dataclasses
from collections.abc import Callable
from typing import TypeVar

from signalflow.enums import ComponentType
from signalflow.registry import registry

T = TypeVar("T", bound=type)


def _make(component_type: ComponentType, role: str) -> Callable[..., Callable[[T], T]]:
    def decorator(name: str, override: bool = False) -> Callable[[T], T]:
        def wrap(cls: T) -> T:
            if component_type is ComponentType.TRANSFORM:
                from signalflow.transform.base import Transform

                if (
                    not dataclasses.is_dataclass(cls)
                    and cls.__init__ is not object.__init__
                    and getattr(cls, "to_config", None) is Transform.to_config
                ):
                    raise TypeError(
                        f"transform {name!r} ({cls.__qualname__}) must be a dataclass so its "
                        f"parameters round-trip flow.yaml; add @dataclass above the class"
                    )
            cls._sf_name = name
            cls._sf_role = role
            cls._sf_type = component_type
            registry.register(component_type, name, cls, role=role, override=override)
            return cls

        return wrap

    return decorator


transform = _make(ComponentType.TRANSFORM, role="transform")
feature = _make(ComponentType.TRANSFORM, role="feature")
detector = _make(ComponentType.TRANSFORM, role="detector")
model = _make(ComponentType.MODEL, role="model")
strategy = _make(ComponentType.STRATEGY, role="strategy")
sampler = _make(ComponentType.SAMPLER, role="sampler")
broker = _make(ComponentType.BROKER, role="broker")
metric = _make(ComponentType.METRIC, role="metric")
source = _make(ComponentType.SOURCE, role="source")
