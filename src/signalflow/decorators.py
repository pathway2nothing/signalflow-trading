"""Semantic registration decorators."""

from collections.abc import Callable
from typing import TypeVar

from signalflow.enums import ComponentType
from signalflow.registry import registry

T = TypeVar("T", bound=type)


def _make(component_type: ComponentType, role: str) -> Callable[..., Callable[[T], T]]:
    def decorator(name: str, override: bool = False) -> Callable[[T], T]:
        def wrap(cls: T) -> T:
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
