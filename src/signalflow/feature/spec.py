"""FeatureSpec — serializable, hashable recipe for a feature pipeline.

A ``FeatureSpec`` captures *how* to rebuild a :class:`FeaturePipeline` (the
recipe), not the computed feature values. It serializes to / from a plain
``dict`` (YAML-friendly) and produces a stable ``feature_hash`` that is:

* identical for two logically-equal specs (key order in ``params`` irrelevant,
  float jitter normalized, defaults resolved), and
* different whenever something *meaningful* changes (a param value, the order
  of features, or the ``ta_version`` pinning the implementation).

The hash is a configuration-drift detector between train and serve: recompute
it when loading a model artifact and refuse to continue on mismatch.

See VISION.md §5.4–5.5 for the design intent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from signalflow.core import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.feature.base import Feature, GlobalFeature
from signalflow.feature.feature_pipeline import FeaturePipeline

# Float canonicalization precision. 12 significant decimals is enough to make
# 0.1 == 0.1000000001 collapse to the same string while keeping genuinely
# different values (e.g. 0.1 vs 0.11) distinct.
_FLOAT_PRECISION = 12

# Infrastructure / base-class fields that are NOT user-facing feature params.
# These are wiring (column names, normalization plumbing) rather than part of
# the feature's logical recipe, so they must be excluded from the hash and the
# serialized params — otherwise an internal rename would falsely change a hash.
# Mirrors SignalFlowRegistry._BASE_FIELDS (feature-relevant subset).
_NON_PARAM_FIELDS: frozenset[str] = frozenset(
    {
        "component_type",
        "group_col",
        "ts_col",
        "normalized",
        "norm_period",
    }
)


def _registry_name_for_class(cls: type) -> str | None:
    """Reverse-lookup a feature's registry name from its class.

    The registry (``SignalFlowRegistry``) is a forward map name -> ComponentInfo
    and exposes **no** class -> name reverse index. We rebuild it on demand by
    scanning ``default_registry`` for the registered class identity. This is
    O(n) over registered features but only runs at spec-construction time.

    Returns the registry name, or ``None`` if the class is not registered (the
    caller then falls back to the class name and we document that limitation).
    """
    default_registry._discover_if_needed()
    items = default_registry._items.get(SfComponentType.FEATURE, {})
    for name, info in items.items():
        if info.cls is cls:
            return name
    return None


def _scope_for_feature(feat: Feature) -> str:
    """Classify a feature instance as global- or pair-scoped."""
    if isinstance(feat, (GlobalFeature, FeaturePipeline)) or getattr(feat, "_is_global", False):
        return "global"
    return "pair"


def _resolved_params(feat: Feature) -> dict[str, Any]:
    """Extract user-facing params with defaults resolved explicitly.

    Uses ``dataclasses.fields()`` so that omitted params (e.g. ``rsi()`` vs
    ``rsi(period=14)``) resolve to the same explicit value before hashing,
    closing the "defaults" false-mismatch trap (VISION.md §5.5).
    """
    params: dict[str, Any] = {}
    for f in dataclasses.fields(feat):
        if f.name.startswith("_") or f.name in _NON_PARAM_FIELDS:
            continue
        params[f.name] = getattr(feat, f.name)
    return params


def _canonicalize(value: Any) -> Any:
    """Recursively canonicalize a value for stable hashing.

    * floats -> rounded to fixed precision (kills representation jitter);
    * dicts -> canonicalized values (key order handled later by sort_keys);
    * lists / tuples -> element-wise canonicalized (order preserved);
    * bool is left as-is (note: bool is a subclass of int, handled before int).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # round to fixed precision; normalize -0.0 to 0.0
        r = round(value, _FLOAT_PRECISION)
        return r + 0.0
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


def canonical_feature_hash(
    features: list[dict],
    ta_version: str | None,
    raw_data_type: str,
) -> str:
    """Compute the stable feature hash from a feature recipe.

    Pure function (no I/O, no global state) — trivially unit-testable.

    Canonicalization rules (VISION.md §5.5):
      * ``json.dumps(..., sort_keys=True)`` makes dict **key** order irrelevant;
      * floats are rounded to fixed precision so 0.1 == 0.1000000001;
      * the **order of features is significant** — the list is hashed as-is,
        never sorted;
      * ``ta_version`` is part of the hash (same feature name across TA library
        versions is not the same implementation).

    Args:
        features: Ordered list of ``{"name", "params", "scope"}`` dicts.
        ta_version: Pinned TA-implementation version, or ``None``.
        raw_data_type: Raw data type key (e.g. ``"spot"``).

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = {
        "features": _canonicalize(features),  # order preserved
        "ta_version": ta_version,
        "raw_data_type": raw_data_type,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class FeatureSpec:
    """Serializable, hashable recipe for a :class:`FeaturePipeline`.

    Attributes:
        features: Ordered feature records. Each is
            ``{"name": str, "params": dict, "scope": "pair"|"global"}``.
            Order is significant and preserved everywhere.
        ta_version: Pinned TA-implementation version (part of the hash).
        raw_data_type: Raw data type key (e.g. ``"spot"``).
        order_significant: Declares that feature order is part of the truth.
            Always honored (features are never reordered); kept as serialized
            metadata for provenance.
    """

    features: list[dict] = field(default_factory=list)
    ta_version: str | None = None
    raw_data_type: str = "spot"
    order_significant: bool = True

    # ── Construction from a live pipeline ────────────────────────────────

    @classmethod
    def from_pipeline(
        cls,
        pipeline: FeaturePipeline,
        *,
        ta_version: str | None = None,
    ) -> FeatureSpec:
        """Extract a :class:`FeatureSpec` from a live pipeline.

        Feature names come from the registry reverse-lookup
        (:func:`_registry_name_for_class`). If a feature class is not
        registered, we fall back to its class name and the resulting spec may
        not round-trip through :meth:`build` (documented limitation).
        """
        records: list[dict] = []
        for feat in pipeline.features:
            name = _registry_name_for_class(type(feat))
            if name is None:
                # Fallback: unregistered class. build() would fail for this
                # name, but we still produce a stable, hashable record.
                name = type(feat).__name__
            records.append(
                {
                    "name": name,
                    "params": _resolved_params(feat),
                    "scope": _scope_for_feature(feat),
                }
            )

        raw = getattr(pipeline.raw_data_type, "value", pipeline.raw_data_type)
        return cls(
            features=records,
            ta_version=ta_version,
            raw_data_type=str(raw),
        )

    # ── Config (dict) round-trip ─────────────────────────────────────────

    @classmethod
    def from_config(cls, data: dict) -> FeatureSpec:
        """Reconstruct a spec from a plain config dict.

        Accepts both the flat form (``ta_version``/``raw_data_type`` at top
        level) and the YAML ``meta:`` nested form shown in VISION.md §5.4.
        """
        meta = data.get("meta", {}) or {}
        features_in = data.get("features", []) or []

        features: list[dict] = []
        for rec in features_in:
            features.append(
                {
                    "name": rec["name"],
                    "params": dict(rec.get("params", {}) or {}),
                    "scope": rec.get("scope", "pair"),
                }
            )

        ta_version = data.get("ta_version", meta.get("ta_version"))
        raw_data_type = data.get("raw_data_type", meta.get("raw_data_type", "spot"))
        order_significant = data.get("order_significant", meta.get("order_significant", True))

        return cls(
            features=features,
            ta_version=ta_version,
            raw_data_type=str(raw_data_type),
            order_significant=bool(order_significant),
        )

    def to_config(self) -> dict:
        """Serialize to a plain dict (YAML ``meta:`` nested form)."""
        return {
            "features": [
                {
                    "name": rec["name"],
                    "params": dict(rec.get("params", {}) or {}),
                    "scope": rec.get("scope", "pair"),
                }
                for rec in self.features
            ],
            "meta": {
                "ta_version": self.ta_version,
                "raw_data_type": self.raw_data_type,
                "order_significant": self.order_significant,
            },
        }

    # ── Reconstruction ───────────────────────────────────────────────────

    def build(self) -> FeaturePipeline:
        """Reconstruct a :class:`FeaturePipeline` from the recipe.

        For each record, instantiate via
        ``default_registry.create(FEATURE, name, **params)`` and assemble in the
        **given order** (order is significant).
        """
        feats: list[Feature] = [
            default_registry.create(
                SfComponentType.FEATURE,
                rec["name"],
                **dict(rec.get("params", {}) or {}),
            )
            for rec in self.features
        ]
        return FeaturePipeline(features=feats, raw_data_type=self.raw_data_type)

    # ── YAML round-trip ───────────────────────────────────────────────────

    def to_yaml(self, path: str | Path) -> None:
        """Persist the spec as YAML (survives class refactors, unlike pickle)."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_config(), fh, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureSpec:
        """Load a spec from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_config(data)

    # ── Hashing ────────────────────────────────────────────────────────────

    def feature_hash(self) -> str:
        """Stable SHA-256 of the canonical recipe (drift detector)."""
        return canonical_feature_hash(
            self.features,
            self.ta_version,
            self.raw_data_type,
        )
