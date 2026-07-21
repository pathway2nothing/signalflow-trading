"""
The Transform contract - the single computation primitive.

DataFrame in → columns out, with declared ``warmup`` and ``outputs``, causal by
construction. Features, detectors, encoders and signal meta-features are all
Transforms; *role* (feature vs detector) is set by the decorator and by which
slot a transform occupies, not by a separate base class.

Two categories:

* **stateless** (``requires_fit=False``) - pure ``compute``; e.g. RSI, ATR.
* **stateful** (``requires_fit=True``) - must ``fit`` on a past window before
  ``compute``; ``requires_target=True`` additionally needs labels (e.g. WoE).
  Stateful transforms are fit *inside* the walk-forward split by the model, so a
  scaler can never see the future.

Causality is enforced for :class:`Feature`: you return Polars expressions and the
base applies them ``.over("pair")`` on a ts-sorted frame, so a rolling window
physically cannot cross a pair boundary or see past ``t``.
"""

import dataclasses
from abc import ABC, abstractmethod

import polars as pl

from signalflow.errors import PipeError, UnfittedTransformError


class Transform(ABC):
    """Base contract for every computation in the framework.

    A parameterized subclass must be a ``@dataclass`` (or override ``to_config``)
    so its constructor parameters round-trip through flow.yaml.
    """

    requires_fit: bool = False
    requires_target: bool = False

    @property
    def name(self) -> str:
        return getattr(self, "_sf_name", type(self).__name__)

    @property
    def role(self) -> str:
        return getattr(self, "_sf_role", "transform")

    def to_config(self) -> dict:
        """Round-trippable ``{transform, role, params}`` (registry reconstructs it).

        Subclasses carrying constructor parameters must be dataclasses so those
        parameters can be captured for the flow.yaml round-trip.
        """
        if not dataclasses.is_dataclass(self) and type(self).__init__ is not object.__init__:
            raise PipeError(
                f"{type(self).__qualname__} is not a dataclass; its constructor parameters "
                f"cannot be captured for flow.yaml round-trip"
            )
        params = {}
        if dataclasses.is_dataclass(self):
            for f in dataclasses.fields(self):
                if f.name.startswith("_"):
                    continue
                val = getattr(self, f.name)
                params[f.name] = val.to_config() if isinstance(val, Transform) else val
        return {"transform": self.name, "role": self.role, "params": params}

    @classmethod
    def from_config(cls, cfg: dict) -> "Transform":
        """Inverse of :meth:`to_config`, rebuilding nested transforms recursively."""
        params = {k: _rebuild_value(v) for k, v in (cfg.get("params") or {}).items()}
        return cls(**params)

    @property
    def warmup(self) -> int:
        return 0

    @property
    @abstractmethod
    def outputs(self) -> list[str]:
        """Column names this transform appends."""

    def fit(self, df: pl.DataFrame, target: pl.Series | None = None) -> "Transform":
        """Stateful transforms override; stateless ones are a no-op."""
        return self

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append :pyattr:`outputs` to ``df`` (causal)."""

    def _require_fitted(self, attr: str) -> None:
        if not hasattr(self, attr):
            raise UnfittedTransformError(f"{self.name}: call fit() before compute()")


class Feature(Transform):
    """A causal, stateless feature expressed as Polars expressions."""

    @abstractmethod
    def exprs(self) -> list[pl.Expr]:
        """Expressions whose ``.alias(...)`` names match :pyattr:`outputs`."""

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        sorted_df = df.sort(["pair", "ts"])
        return sorted_df.with_columns([e.over("pair") for e in self.exprs()])


def _is_transform_config(value: object) -> bool:
    return isinstance(value, dict) and "transform" in value


def _rebuild_value(value: object) -> object:
    if _is_transform_config(value):
        return build_transform(value)
    if isinstance(value, list):
        return [build_transform(v) if _is_transform_config(v) else v for v in value]
    return value


def build_transform(cfg: dict) -> Transform:
    """Reconstruct any registered transform (recursively) from its ``to_config``."""
    from signalflow.enums import ComponentType
    from signalflow.registry import registry

    cls = registry.get(ComponentType.TRANSFORM, cfg["transform"])
    return cls.from_config(cfg)
