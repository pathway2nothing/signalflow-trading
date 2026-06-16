"""Rolling WoE refit (refit/window effective) + fitted-state serialization."""

from datetime import UTC, datetime, timedelta

import polars as pl

import signalflow as sf
from signalflow.model.oos import parse_duration, rolling_folds
from signalflow.transform.encode.woe import Binning, WoE


def test_parse_duration():
    assert parse_duration("1d") == timedelta(days=1)
    assert parse_duration("365d") == timedelta(days=365)
    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("30m") == timedelta(minutes=30)


def test_rolling_folds_step_and_window():
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(24 * 10)]
    folds = rolling_folds(ts, parse_duration("1d"), parse_duration("3d"), timedelta(hours=6))
    assert len(folds) >= 8
    assert all(f.train_start_ts is not None for f in folds)
    assert folds[1].test_start_ts - folds[0].test_start_ts == timedelta(days=1)
    f = folds[3]
    assert f.train_end_ts - f.train_start_ts == timedelta(days=3)


def test_woe_state_round_trip():
    df = pl.DataFrame({"f1": [float(i % 17) for i in range(300)], "f2": [float(i % 23) for i in range(300)]})
    target = pl.Series([1.0 if i % 3 == 0 else -1.0 for i in range(300)])
    woe = WoE(binning=Binning("quantile", 5), columns=["f1", "f2"]).fit(df, target)
    state = woe.state_dict()
    assert set(state["edges"]) == {"f1", "f2"}
    fresh = WoE(binning=Binning("quantile", 5), columns=["f1", "f2"]).load_state(state)
    assert fresh.compute(df).equals(woe.compute(df))


def test_model_records_rolling_refits():
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-12", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=6),
        features=sf.FeaturePipe(sf.SMA(5), sf.SMA(10)),
        encode=WoE(refit="1d", window="3d"),
        output="p_rise",
        min_train_rows=20,
    )
    model.fit(ds)
    hist = model.woe_history()
    assert len(hist) >= 3
    starts = [h["test_start"] for h in hist]
    assert starts == sorted(starts)
    assert all(h["train_start"] is not None for h in hist)
    assert "edges" in hist[0]["state"] and hist[0]["state"]["columns"]
    assert hist[0]["target"] and hist[0]["state"]["binning"]


def _refit_model():
    return sf.ForecastModel(
        target=sf.FixedHorizon(bars=6),
        features=sf.FeaturePipe(sf.SMA(5)),
        encode=WoE(refit="1d", window="3d"),
        min_train_rows=20,
    )


def test_fold_cache_reuse_skips_recompute(tmp_path):
    from signalflow.experiment.cache import ArtifactCache

    ds = sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-12", interval="1h")
    cache = ArtifactCache(str(tmp_path / "folds"))

    m1 = _refit_model().fit(ds, cache=cache)
    assert any(cache.root.iterdir())

    m2 = _refit_model()

    def _boom(*_a, **_k):
        raise AssertionError("recomputed a fold that should be cached")

    m2._fit_fold_predict = _boom
    m2.fit(ds, cache=cache)

    assert m1.oos_.equals(m2.oos_)
    assert len(m1.woe_history()) == len(m2.woe_history())


def test_model_dump_woe_history(tmp_path):
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-10", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=6),
        features=sf.FeaturePipe(sf.SMA(5)),
        encode=WoE(refit="1d", window="3d"),
        min_train_rows=20,
    )
    model.fit(ds)
    path = str(tmp_path / "woe_history.json")
    model.dump_woe_history(path)
    import json

    with open(path) as fh:
        loaded = json.load(fh)
    assert isinstance(loaded, list) and loaded and "state" in loaded[0]
