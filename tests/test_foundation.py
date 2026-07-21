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


def test_plugin_api_handshake_warns_and_errors(monkeypatch):
    import importlib
    from types import ModuleType

    from loguru import logger

    reg_mod = importlib.import_module("signalflow.registry")

    mod_missing = ModuleType("fake_plugin_missing")
    mod_wrong = ModuleType("fake_plugin_wrong")
    mod_wrong.SIGNALFLOW_PLUGIN_API = 999

    class _EP:
        def __init__(self, name, module):
            self.name = name
            self._module = module

        def load(self):
            return self._module

    class _EPS:
        def select(self, group=None):
            return [_EP("missingplugin", mod_missing), _EP("wrongplugin", mod_wrong)]

    monkeypatch.setattr(reg_mod, "entry_points", lambda: _EPS())

    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        reg_mod.Registry()._discover_entry_points()
    finally:
        logger.remove(sink)
    text = "\n".join(msgs)
    assert "does not declare SIGNALFLOW_PLUGIN_API" in text
    assert "targets plugin API 999" in text


def test_register_detector_export():
    from dataclasses import dataclass

    for name in (
        "register_transform",
        "register_feature",
        "register_detector",
        "register_model",
        "register_strategy",
        "register_sampler",
        "register_broker",
        "register_metric",
        "register_source",
    ):
        assert callable(getattr(sf, name))

    @sf.register_detector("d07_export_test")
    @dataclass
    class D07Above(sf.SignalDetector):
        n: int = 10

        @property
        def warmup(self) -> int:
            return self.n

        def detect(self, df):
            sma = pl.col("close").rolling_mean(self.n).over("pair")
            return df.with_columns(
                pl.when(pl.col("close") > sma).then(pl.lit(sf.RISE)).otherwise(pl.lit(sf.NONE)).alias("signal")
            )

    assert sf.registry.get(sf.ComponentType.TRANSFORM, "d07_export_test") is D07Above


def test_registry_error_is_signalflow_error():
    from signalflow.decorators import feature

    assert issubclass(sf.RegistryError, sf.SignalFlowError)
    assert issubclass(sf.RegistryError, ValueError)

    @feature("d06_1_first")
    class D06First:
        pass

    with pytest.raises(sf.RegistryError):

        @feature("d06_1_first")
        class D06Second:
            pass


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
