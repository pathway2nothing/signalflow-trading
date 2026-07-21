"""FeatureStore: content-addressed reuse of computed feature frames."""

import warnings

import pytest

import signalflow as sf
from signalflow.transform.pipe import FeaturePipe

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")


def _ds():
    return sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-03-01", interval="1h")


def _count_compute(monkeypatch):
    calls = {"n": 0}
    orig = FeaturePipe.compute

    def counting(self, df):
        calls["n"] += 1
        return orig(self, df)

    monkeypatch.setattr(FeaturePipe, "compute", counting)
    return calls


def test_second_fit_skips_recompute(tmp_path, monkeypatch):
    calls = _count_compute(monkeypatch)
    ds = _ds()
    store = sf.FeatureStore(tmp_path)
    sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)), encode=None, select=None).fit(
        ds, feature_store=store
    )
    after_first = calls["n"]
    assert after_first >= 1
    sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)), encode=None, select=None).fit(
        ds, feature_store=store
    )
    assert calls["n"] == after_first


def test_param_change_is_cache_miss(tmp_path, monkeypatch):
    calls = _count_compute(monkeypatch)
    ds = _ds()
    store = sf.FeatureStore(tmp_path)
    sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(10)), encode=None, select=None).fit(
        ds, feature_store=store
    )
    n1 = calls["n"]
    sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(11)), encode=None, select=None).fit(
        ds, feature_store=store
    )
    assert calls["n"] > n1


def test_span_change_is_cache_miss(tmp_path):
    store = sf.FeatureStore(tmp_path)
    pipe = sf.FeaturePipe(sf.SMA(20))
    a = _ds()
    b = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-04-01", interval="1h")
    assert store.key(pipe, a) != store.key(pipe, b)


def test_stored_frame_equals_fresh(tmp_path):
    ds = _ds()
    store = sf.FeatureStore(tmp_path)
    pipe = sf.FeaturePipe(sf.SMA(20))
    stored = store.compute(pipe, ds)
    fresh = pipe.compute(ds.frame)
    again = store.compute(pipe, ds)
    assert stored.equals(fresh)
    assert again.equals(fresh)
