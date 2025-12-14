from dataclasses import dataclass, field
from typing import Any, Dict, Type
from loguru import logger
from .enums import SfComponentType


@dataclass
class SignalFlowRegistry:
    """Component registry.

    Stores mapping: component_type -> name -> class.
    """
    #TODO: Registry autodiscover

    _items: Dict[SfComponentType, Dict[str, Type[Any]]] = field(default_factory=dict)

    def _ensure(self, component_type: SfComponentType) -> None:
        self._items.setdefault(component_type, {})

    def register(self, component_type: SfComponentType, name: str, cls: Type[Any], *, override: bool = False) -> None:
        """Register a class under (component_type, name)."""
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
        """Get a registered class by key."""
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
        """Instantiate a component by registry key."""
        cls = self.get(component_type, name)
        return cls(**kwargs)

    def list(self, component_type: SfComponentType) -> list[str]:
        """List registered components for a type."""
        self._ensure(component_type)
        return sorted(self._items[component_type])

    def snapshot(self) -> dict[str, list[str]]:
        """Snapshot of registry for debugging."""
        return {t.value: sorted(v.keys()) for t, v in self._items.items()}


default_registry = SignalFlowRegistry()