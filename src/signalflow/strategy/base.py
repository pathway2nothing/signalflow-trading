"""Declarative strategy config contract mirroring the transform round-trip."""

import dataclasses


def _encode(value: object) -> object:
    if isinstance(value, Strategy):
        return value.to_config()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value) if not f.name.startswith("_")
        }
    return value


def _is_strategy_config(value: object) -> bool:
    return isinstance(value, dict) and "name" in value and "params" in value


def _rebuild_value(ftype: object, value: object) -> object:
    if _is_strategy_config(value):
        return build_strategy(value)
    if isinstance(ftype, type) and dataclasses.is_dataclass(ftype) and isinstance(value, dict):
        return ftype(**value)
    return value


class Strategy:
    """Base mixin giving strategies the same config round-trip transforms have."""

    @property
    def name(self) -> str:
        return getattr(self, "_sf_name", type(self).__name__)

    def to_config(self) -> dict:
        """Round-trippable ``{name, params}`` the registry reconstructs."""
        params: dict = {}
        if dataclasses.is_dataclass(self):
            for f in dataclasses.fields(self):
                if f.name.startswith("_") or not f.init:
                    continue
                params[f.name] = _encode(getattr(self, f.name))
        return {"name": self.name, "params": params}

    @classmethod
    def from_config(cls, cfg: dict) -> "Strategy":
        """Inverse of :meth:`to_config`, rebuilding nested strategies and dataclasses."""
        field_types = {f.name: f.type for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
        params = {k: _rebuild_value(field_types.get(k), v) for k, v in (cfg.get("params") or {}).items()}
        return cls(**params)


def build_strategy(cfg: dict) -> Strategy:
    """Reconstruct any registered strategy from its ``to_config`` (legacy shapes tolerated)."""
    from signalflow.enums import ComponentType
    from signalflow.registry import registry

    name = cfg.get("name", "rules")
    cls = registry.get(ComponentType.STRATEGY, name)
    if "params" not in cfg and ("entry" in cfg or "exit" in cfg):
        cfg = {"name": name, "params": {"entry": cfg.get("entry", {}), "exit": cfg.get("exit", {})}}
    return cls.from_config(cfg)
