"""Declarative round-trip: to_config <-> build_transform for pipes and detectors."""

import cloudpickle
import polars as pl

import signalflow as sf
from signalflow.transform.base import build_transform
from signalflow.transform.encode.woe import Binning, WoE


def _ds():
    return sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-05", interval="1h")


def test_feature_pipe_config_round_trip():
    pipe = sf.FeaturePipe(sf.SMA(10), sf.SMA(20))
    rebuilt = build_transform(pipe.to_config())
    assert isinstance(rebuilt, sf.FeaturePipe)
    assert rebuilt.outputs == pipe.outputs
    ds = _ds()
    assert rebuilt.compute(ds.frame).equals(pipe.compute(ds.frame))


def test_nested_feature_pipe_round_trip():
    pipe = sf.FeaturePipe(sf.FeaturePipe(sf.SMA(5)), sf.SMA(30))
    rebuilt = build_transform(pipe.to_config())
    assert rebuilt.outputs == pipe.outputs == ["sma_5", "sma_30"]
    inner = rebuilt.transforms[0]
    assert isinstance(inner, sf.FeaturePipe)


def test_feature_pipe_save_load(tmp_path):
    pipe = sf.FeaturePipe(sf.SMA(10), sf.SMA(50))
    path = str(tmp_path / "pipe.yaml")
    pipe.save(path)
    loaded = sf.FeaturePipe.load(path)
    assert loaded.outputs == pipe.outputs


def test_detector_config_round_trip():
    det = sf.SmaCrossDetector(fast=5, slow=20)
    rebuilt = build_transform(det.to_config())
    assert rebuilt.to_config() == det.to_config()


def test_flow_yaml_detector_round_trip(tmp_path):
    flow = sf.Flow(name="ser", detectors=[sf.SmaCrossDetector(fast=3, slow=8)])
    path = str(tmp_path / "flow.yaml")
    flow.save(path)
    loaded = sf.Flow.load(path)
    assert [d.to_config() for d in loaded.detectors] == [d.to_config() for d in flow.detectors]


def test_woe_recipe_round_trip():
    woe = WoE(binning=Binning("quantile", 8), smoothing=0.7, columns=["a", "b"])
    rebuilt = build_transform(woe.to_config())
    assert isinstance(rebuilt, WoE)
    assert isinstance(rebuilt.binning, Binning)
    assert rebuilt.binning.method == "quantile" and rebuilt.binning.max_bins == 8
    assert rebuilt.smoothing == 0.7 and rebuilt.columns == ["a", "b"]
    assert rebuilt.to_config() == woe.to_config()


def test_woe_fitted_state_survives_pickle():
    n = 300
    df = pl.DataFrame({"f1": [float(i % 17) for i in range(n)], "f2": [float(i % 23) for i in range(n)]})
    target = pl.Series([1.0 if i % 3 == 0 else -1.0 for i in range(n)])
    woe = WoE(binning=Binning("quantile", 5), columns=["f1", "f2"]).fit(df, target)
    back = cloudpickle.loads(cloudpickle.dumps(woe))
    assert back.outputs == woe.outputs
    assert back.compute(df).equals(woe.compute(df))
