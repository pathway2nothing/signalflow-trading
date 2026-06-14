"""
Central component registry - the backbone of serialization.

Every core class registers under a name; that name is what ``flow.yaml``
serializes, ``sf list`` enumerates, and plugin packages inject into. Construction
from config is ``registry.create(ComponentType.TRANSFORM, "revert_confluence",
**params)``-shaped. Seven types instead of the old 21.

The design (lazy autodiscovery, dataclass-field schema introspection) is the
proven one from the previous framework, trimmed to the current type set.
"""


import dataclasses
import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from loguru import logger

from signalflow.enums import ComponentType
from signalflow.errors import UnknownComponentError


@dataclass
class ComponentInfo:
    """Metadata about a registered component (class + extracted docs + role)."""

    cls: type[Any]
    role: str = ""
    docstring: str = ""
    summary: str = ""
    module: str = ""

    @classmethod
    def from_class(cls, component_cls: type[Any], role: str = "") -> "ComponentInfo":
        doc = (component_cls.__doc__ or "").strip()
        summary = doc.split("\n", 1)[0] if doc else ""
        return cls(
            cls=component_cls,
            role=role or getattr(component_cls, "_sf_role", ""),
            docstring=doc,
            summary=summary,
            module=getattr(component_cls, "__module__", ""),
        )


@dataclass
class Registry:
    """Maps ``ComponentType -> name -> ComponentInfo`` with lazy autodiscovery."""

    _items: dict[ComponentType, dict[str, ComponentInfo]] = field(default_factory=dict)
    _discovered: bool = field(default=False, repr=False)


    def register(
        self,
        component_type: ComponentType,
        name: str,
        cls: type[Any],
        *,
        role: str = "",
        override: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        key = name.strip().lower()
        bucket = self._items.setdefault(component_type, {})
        if key in bucket and not override:
            raise ValueError(f"{component_type.value}:{key} already registered")
        if key in bucket and override:
            logger.warning(f"Overriding {component_type.value}:{key} with {cls.__name__}")
        bucket[key] = ComponentInfo.from_class(cls, role=role)


    def get(self, component_type: ComponentType, name: str) -> type[Any]:
        self._discover_if_needed()
        bucket = self._items.get(component_type, {})
        key = name.lower()
        try:
            return bucket[key].cls
        except KeyError as e:
            available = ", ".join(sorted(bucket))
            raise UnknownComponentError(
                f"{component_type.value}:{key} not found. Available: [{available}]"
            ) from e

    def get_info(self, component_type: ComponentType, name: str) -> ComponentInfo:
        self._discover_if_needed()
        bucket = self._items.get(component_type, {})
        key = name.lower()
        try:
            return bucket[key]
        except KeyError as e:
            available = ", ".join(sorted(bucket))
            raise UnknownComponentError(
                f"{component_type.value}:{key} not found. Available: [{available}]"
            ) from e

    def create(self, component_type: ComponentType, name: str, **kwargs: Any) -> Any:
        return self.get(component_type, name)(**kwargs)

    def list(self, component_type: ComponentType) -> list[str]:
        self._discover_if_needed()
        return sorted(self._items.get(component_type, {}))

    def snapshot(self) -> "dict[str, list[str]]":
        self._discover_if_needed()
        return {t.value: sorted(v) for t, v in self._items.items()}


    def get_schema(self, component_type: ComponentType, name: str) -> dict[str, Any]:
        info = self.get_info(component_type, name)
        cls = info.cls
        params: list[dict[str, Any]] = []
        if dataclasses.is_dataclass(cls):
            for f in dataclasses.fields(cls):
                if f.name.startswith("_"):
                    continue
                type_str = self._type_to_str(f.type)
                if "ClassVar" in type_str:
                    continue
                has_default = f.default is not dataclasses.MISSING
                has_factory = f.default_factory is not dataclasses.MISSING
                params.append(
                    {
                        "name": f.name,
                        "type": type_str,
                        "default": f.default if has_default else None,
                        "required": not has_default and not has_factory,
                    }
                )
        return {
            "name": name,
            "class_name": cls.__name__,
            "component_type": component_type.value,
            "role": info.role,
            "description": info.summary,
            "module": info.module,
            "parameters": params,
        }

    @staticmethod
    def _type_to_str(annotation: Any) -> str:
        s = str(annotation)
        for prefix in ("typing.", "signalflow."):
            s = s.replace(prefix, "")
        if s.startswith("<class '") and s.endswith("'>"):
            s = s[8:-2]
        return s


    def _discover_if_needed(self) -> None:
        if not self._discovered:
            self.autodiscover()

    def autodiscover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        self._discover_internal()
        self._discover_entry_points()

    def _discover_internal(self) -> None:
        try:
            import signalflow as _root
        except ImportError:
            return
        pkg_path = getattr(_root, "__path__", None)
        if pkg_path is None:
            return
        for _imp, modname, _is_pkg in pkgutil.walk_packages(pkg_path, prefix="signalflow."):
            if modname in sys.modules:
                continue
            try:
                importlib.import_module(modname)
            except Exception as e:
                logger.debug(f"autodiscover: skip {modname}: {e}")

    def _discover_entry_points(self) -> None:
        eps = entry_points()
        group = eps.select(group="signalflow.components") if hasattr(eps, "select") else []
        for ep in group:
            try:
                ep.load()
            except Exception as e:
                logger.warning(f"autodiscover: failed entry-point {ep.name!r}: {e}")


registry = Registry()
"""Global default registry singleton."""
