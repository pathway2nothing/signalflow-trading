"""Tests for signalflow.feature.spec.FeatureSpec and canonical_feature_hash.

Uses real registered feature names (verified via
``default_registry.list(SfComponentType.FEATURE)``): ``example/sma``,
``example/rsi``, ``atr``.
"""

from __future__ import annotations

from signalflow.core import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.feature.examples import ExampleRsiFeature, ExampleSmaFeature
from signalflow.feature.feature_pipeline import FeaturePipeline
from signalflow.feature.spec import (
    FeatureSpec,
    canonical_feature_hash,
)

# Real registry names.
SMA = "example/sma"
RSI = "example/rsi"


def _spec_two() -> FeatureSpec:
    return FeatureSpec(
        features=[
            {"name": SMA, "params": {"period": 20, "price_col": "close"}, "scope": "pair"},
            {"name": RSI, "params": {"period": 14, "price_col": "close"}, "scope": "pair"},
        ],
        ta_version="0.4.2",
        raw_data_type="spot",
    )


def test_registered_names_present():
    """Sanity: the names we use are really registered."""
    names = default_registry.list(SfComponentType.FEATURE)
    assert SMA in names
    assert RSI in names


# ── Determinism ──────────────────────────────────────────────────────────


def test_same_spec_same_hash():
    assert _spec_two().feature_hash() == _spec_two().feature_hash()


def test_pure_function_deterministic():
    feats = [{"name": SMA, "params": {"period": 20}, "scope": "pair"}]
    h1 = canonical_feature_hash(feats, "0.4.2", "spot")
    h2 = canonical_feature_hash(feats, "0.4.2", "spot")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


# ── Significant changes -> different hash ──────────────────────────────────


def test_param_change_changes_hash():
    base = _spec_two().feature_hash()
    changed = FeatureSpec(
        features=[
            {"name": SMA, "params": {"period": 21, "price_col": "close"}, "scope": "pair"},
            {"name": RSI, "params": {"period": 14, "price_col": "close"}, "scope": "pair"},
        ],
        ta_version="0.4.2",
        raw_data_type="spot",
    ).feature_hash()
    assert base != changed


def test_feature_order_changes_hash():
    base = _spec_two().feature_hash()
    reordered = FeatureSpec(
        features=[
            {"name": RSI, "params": {"period": 14, "price_col": "close"}, "scope": "pair"},
            {"name": SMA, "params": {"period": 20, "price_col": "close"}, "scope": "pair"},
        ],
        ta_version="0.4.2",
        raw_data_type="spot",
    ).feature_hash()
    assert base != reordered


def test_ta_version_changes_hash():
    base = _spec_two().feature_hash()
    other = _spec_two()
    other.ta_version = "0.5.0"
    assert base != other.feature_hash()


def test_raw_data_type_changes_hash():
    base = _spec_two().feature_hash()
    other = _spec_two()
    other.raw_data_type = "futures"
    assert base != other.feature_hash()


# ── False-mismatch traps closed ────────────────────────────────────────────


def test_param_key_order_irrelevant():
    """Same params, different dict insertion order -> same hash."""
    a = FeatureSpec(
        features=[{"name": SMA, "params": {"period": 20, "price_col": "close"}, "scope": "pair"}],
        ta_version="0.4.2",
    )
    b = FeatureSpec(
        features=[{"name": SMA, "params": {"price_col": "close", "period": 20}, "scope": "pair"}],
        ta_version="0.4.2",
    )
    assert a.feature_hash() == b.feature_hash()


def test_float_jitter_irrelevant():
    """0.1 and 0.1000000000001 collapse to the same hash."""
    a = canonical_feature_hash([{"name": SMA, "params": {"alpha": 0.1}, "scope": "pair"}], None, "spot")
    b = canonical_feature_hash([{"name": SMA, "params": {"alpha": 0.1000000000001}, "scope": "pair"}], None, "spot")
    assert a == b


def test_genuinely_different_float_differs():
    a = canonical_feature_hash([{"name": SMA, "params": {"alpha": 0.1}, "scope": "pair"}], None, "spot")
    b = canonical_feature_hash([{"name": SMA, "params": {"alpha": 0.11}, "scope": "pair"}], None, "spot")
    assert a != b


# ── Config round-trip ──────────────────────────────────────────────────────


def test_config_round_trip_equal():
    spec = _spec_two()
    cfg = spec.to_config()
    restored = FeatureSpec.from_config(cfg)
    assert restored == spec
    assert restored.feature_hash() == spec.feature_hash()


def test_from_config_flat_form():
    """from_config accepts top-level ta_version/raw_data_type too."""
    cfg = {
        "features": [{"name": SMA, "params": {"period": 20, "price_col": "close"}, "scope": "pair"}],
        "ta_version": "0.4.2",
        "raw_data_type": "spot",
    }
    spec = FeatureSpec.from_config(cfg)
    assert spec.ta_version == "0.4.2"
    assert spec.raw_data_type == "spot"
    assert spec.features[0]["name"] == SMA


# ── build() reconstruction ────────────────────────────────────────────────


def test_build_reconstructs_pipeline_order_and_count():
    spec = _spec_two()
    pipe = spec.build()
    assert isinstance(pipe, FeaturePipeline)
    assert len(pipe.features) == len(spec.features)
    # order preserved: sma then rsi
    assert pipe.features[0].output_cols() == ["sma_20"]
    assert pipe.features[1].output_cols() == ["rsi_14"]


# ── from_pipeline ──────────────────────────────────────────────────────────


def test_from_pipeline_extracts_names_and_params():
    pipe = FeaturePipeline(
        features=[ExampleSmaFeature(period=20), ExampleRsiFeature(period=14)],
        raw_data_type="spot",
    )
    spec = FeatureSpec.from_pipeline(pipe, ta_version="0.4.2")
    assert [r["name"] for r in spec.features] == [SMA, RSI]
    assert spec.features[0]["params"]["period"] == 20
    assert spec.features[1]["params"]["period"] == 14
    assert spec.ta_version == "0.4.2"
    assert spec.raw_data_type == "spot"


def test_from_pipeline_then_build_is_stable():
    """from_pipeline -> build -> from_pipeline gives the same hash."""
    pipe = FeaturePipeline(
        features=[ExampleSmaFeature(period=20), ExampleRsiFeature(period=14)],
        raw_data_type="spot",
    )
    spec1 = FeatureSpec.from_pipeline(pipe, ta_version="0.4.2")
    rebuilt = spec1.build()
    spec2 = FeatureSpec.from_pipeline(rebuilt, ta_version="0.4.2")
    assert spec1.feature_hash() == spec2.feature_hash()


def test_defaults_resolved_in_hash():
    """rsi() and rsi(period=14) must hash the same (defaults resolved)."""
    pipe_default = FeaturePipeline(features=[ExampleRsiFeature()], raw_data_type="spot")
    pipe_explicit = FeaturePipeline(features=[ExampleRsiFeature(period=14)], raw_data_type="spot")
    h1 = FeatureSpec.from_pipeline(pipe_default).feature_hash()
    h2 = FeatureSpec.from_pipeline(pipe_explicit).feature_hash()
    assert h1 == h2


# ── YAML round-trip ────────────────────────────────────────────────────────


def test_yaml_round_trip(tmp_path):
    spec = _spec_two()
    path = tmp_path / "spec.yaml"
    spec.to_yaml(path)
    assert path.exists()
    restored = FeatureSpec.from_yaml(path)
    assert restored == spec
    assert restored.feature_hash() == spec.feature_hash()
