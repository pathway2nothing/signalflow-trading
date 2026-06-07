"""Tests for signalflow.feature.model_features.ModelFeaturesPipeline.

ModelFeaturesPipeline is the reproducibility layer over the single compute
engine (FeaturePipeline). These tests prove:

* zero compute duplication — compute() is byte-identical to the engine's;
* config round-trip and a stable feature_hash matching the underlying spec;
* verify_hash detects recipe drift (loud RuntimeError, never silent);
* validate_reproducible honours the warmup-invariance contract.

Uses real registered feature names (verified in Stage 4): ``example/sma``,
``example/rsi``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import polars as pl
import pytest

from signalflow.core import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.feature.base import Feature
from signalflow.feature.examples import ExampleRsiFeature, ExampleSmaFeature
from signalflow.feature.feature_pipeline import FeaturePipeline
from signalflow.feature.model_features import ModelFeaturesPipeline

SMA = "example/sma"
RSI = "example/rsi"


def _config() -> dict:
    return {
        "features": [
            {"name": SMA, "params": {"period": 20, "price_col": "close"}, "scope": "pair"},
            {"name": RSI, "params": {"period": 14, "price_col": "close"}, "scope": "pair"},
        ],
        "ta_version": "0.4.2",
        "raw_data_type": "spot",
    }


def _make_df(n: int = 120) -> pl.DataFrame:
    rows = []
    for pair in ("BTCUSDT", "ETHUSDT"):
        base = 100.0 if pair == "BTCUSDT" else 50.0
        for i in range(n):
            rows.append({"pair": pair, "timestamp": i, "close": base + i * 0.5})
    return pl.DataFrame(rows)


# ── Sanity ────────────────────────────────────────────────────────────────


def test_registered_names_present():
    names = default_registry.list(SfComponentType.FEATURE)
    assert SMA in names
    assert RSI in names


# ── Zero compute duplication ────────────────────────────────────────────────


def test_compute_identical_to_direct_pipeline():
    """ModelFeaturesPipeline.compute == FeaturePipeline.compute on same data."""
    df = _make_df()

    mfp = ModelFeaturesPipeline.from_config(_config())
    direct = FeatureSpec_build_pipeline()

    out_mfp = mfp.compute(df.clone())
    out_direct = direct.compute(df.clone())

    assert out_mfp.equals(out_direct)


def FeatureSpec_build_pipeline() -> FeaturePipeline:
    """A directly-built engine with the same recipe (no wrapper)."""
    from signalflow.feature.spec import FeatureSpec

    return FeatureSpec.from_config(_config()).build()


# ── Config round-trip + hash ────────────────────────────────────────────────


def test_from_config_to_config_round_trip():
    mfp = ModelFeaturesPipeline.from_config(_config())
    restored = ModelFeaturesPipeline.from_config(mfp.to_config())
    assert restored.to_config() == mfp.to_config()
    assert restored.feature_hash == mfp.feature_hash


def test_feature_hash_stable():
    mfp = ModelFeaturesPipeline.from_config(_config())
    assert mfp.feature_hash == mfp.feature_hash


def test_feature_hash_matches_spec():
    from signalflow.feature.spec import FeatureSpec

    mfp = ModelFeaturesPipeline.from_config(_config())
    spec = FeatureSpec.from_config(_config())
    assert mfp.feature_hash == spec.feature_hash()


def test_from_pipeline_keeps_engine_and_recipe():
    pipe = FeaturePipeline(
        features=[ExampleSmaFeature(period=20), ExampleRsiFeature(period=14)],
        raw_data_type="spot",
    )
    mfp = ModelFeaturesPipeline.from_pipeline(pipe, ta_version="0.4.2")
    # same underlying engine object is reused (composition, not rebuild)
    assert mfp._pipeline is pipe
    assert mfp._spec.ta_version == "0.4.2"
    assert [r["name"] for r in mfp._spec.features] == [SMA, RSI]


def test_to_artifact_dict_bundles_hash():
    mfp = ModelFeaturesPipeline.from_config(_config())
    art = mfp.to_artifact_dict()
    assert art["feature_hash"] == mfp.feature_hash
    assert art["features_config"] == mfp.to_config()


# ── verify_hash drift detection ─────────────────────────────────────────────


def test_verify_hash_matching_ok():
    mfp = ModelFeaturesPipeline.from_config(_config())
    mfp.verify_hash(mfp.feature_hash)  # must not raise


def test_verify_hash_mismatch_raises():
    mfp = ModelFeaturesPipeline.from_config(_config())
    bad = "0" * 64
    with pytest.raises(RuntimeError) as exc:
        mfp.verify_hash(bad)
    msg = str(exc.value)
    # both hashes must be in the message
    assert bad in msg
    assert mfp.feature_hash in msg


# ── validate_reproducible (warmup contract) ──────────────────────────────────


def test_validate_reproducible_ok_for_invariant_features():
    """example/sma + example/rsi are non-recursive → invariant → ok."""
    mfp = ModelFeaturesPipeline.from_config(_config())
    mfp.validate_reproducible()  # must not raise


@dataclass
class _NonInvariantRecursiveFeat(Feature):
    """Recursive, not SMA-seeded → NOT warmup-invariant (breaks parity)."""

    period: int = 10
    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = False
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["nir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("close").alias(f"nir_{self.period}"))


def test_validate_reproducible_raises_for_non_invariant_recursive():
    pipe = FeaturePipeline(
        features=[ExampleSmaFeature(period=20), _NonInvariantRecursiveFeat()],
        raw_data_type="spot",
    )
    mfp = ModelFeaturesPipeline.from_pipeline(pipe)
    with pytest.raises(RuntimeError):
        mfp.validate_reproducible()
