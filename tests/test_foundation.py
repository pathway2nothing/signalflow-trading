"""Foundation: enums, registry, errors, data layer, transforms."""

import polars as pl
import pytest

import signalflow as sf


def test_signal_vocabulary():
    assert sf.RISE == "rise" and sf.FALL == "fall" and sf.NONE == "none"


def test_registry_has_seven_types():
    snap = sf.registry.snapshot()

    assert "transform" in snap
    assert "memory" in sf.registry.list(sf.ComponentType.SOURCE)
    assert "sma" in sf.registry.list(sf.ComponentType.TRANSFORM)
    assert "sma_cross" in sf.registry.list(sf.ComponentType.TRANSFORM)


def test_registry_lists_targets():
    names = sf.registry.list(sf.ComponentType.TARGET)
    assert len(names) >= 26
    for expected in ("fixed_horizon", "triple_barrier", "trend_scanning"):
        assert expected in names


def test_make_target_builds_from_registry():
    from signalflow.target import FixedHorizon, make_target

    t = make_target("fixed_horizon", bars=12)
    assert isinstance(t, FixedHorizon)
    assert t.bars == 12


def test_target_schema_introspection():
    schema = sf.registry.get_schema(sf.ComponentType.TARGET, "triple_barrier_labeler")
    params = {p["name"]: p for p in schema["parameters"]}
    assert params["horizon"]["type"] == "int | str"


def test_registry_unknown_raises():
    with pytest.raises(sf.UnknownComponentError):
        sf.registry.get(sf.ComponentType.TRANSFORM, "does_not_exist")


def test_registry_collision_raises_and_keeps_first():
    from signalflow.decorators import feature

    @feature("spec007_collide")
    class FirstImpl:
        pass

    with pytest.raises(ValueError):

        @feature("spec007_collide")
        class SecondImpl:
            pass

    assert sf.registry.get(sf.ComponentType.TRANSFORM, "spec007_collide") is FirstImpl


def test_registry_override_replaces():
    from signalflow.decorators import feature

    @feature("spec007_override")
    class OrigImpl:
        pass

    @feature("spec007_override", override=True)
    class NewImpl:
        pass

    assert sf.registry.get(sf.ComponentType.TRANSFORM, "spec007_override") is NewImpl


def test_dataset_basics(ds):
    assert set(["BTCUSDT", "ETHUSDT"]) == set(ds.pairs())
    assert ds.height > 0
    bars = list(ds.iter_bars())
    assert len(bars) > 0
    assert "BTCUSDT" in bars[0].prices


def test_cross_rate():
    from signalflow.engine import cross_rate

    assert cross_rate("USDT", "USDT", {}) == 1.0
    assert cross_rate("BTC", "USDT", {"BTCUSDT": 50000.0}) == 50000.0
    assert cross_rate("USDT", "BTC", {"BTCUSDT": 50000.0}) == pytest.approx(1 / 50000.0)


def test_feature_pipe_causal_and_outputs(ds):
    pipe = sf.FeaturePipe(sf.SMA(20), sf.SMA(30), sf.SMA(10))
    out = pipe.compute(ds.frame)
    assert "sma_20" in out.columns and "sma_10" in out.columns
    assert pipe.warmup == 30

    first_eth = out.filter(pl.col("pair") == "ETHUSDT").head(1)
    assert first_eth.get_column("sma_30")[0] is None


def test_feature_pipe_rejects_detector():
    with pytest.raises(sf.PipeError):
        sf.FeaturePipe(sf.SmaCrossDetector(fast=5, slow=10))


def test_transform_roundtrip_config():
    pipe = sf.FeaturePipe(sf.SMA(20), sf.SMA(10))
    cfg = pipe.to_config()
    assert cfg["transform"] == "feature_pipe"
    assert len(cfg["params"]["transforms"]) == 2
