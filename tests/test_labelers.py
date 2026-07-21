"""Tests for the restored legacy labelers, adapted to the Target API."""

import warnings

import polars as pl
import pytest

import signalflow as sf

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def test_import_signalflow():
    pass


def test_import_labelers():
    pass


@pytest.fixture(scope="module")
def small_ds():
    return sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-02-01", interval="1h")


def _labeler_cases():
    from signalflow.target import (
        DirectionalMeanReversionLabeler,
        FixedHorizonLabeler,
        MeanReversionEventLabeler,
        MetaLabelLabeler,
        TripleBarrierLabeler,
        VolatilityRegimeLabeler,
    )

    return [
        (TripleBarrierLabeler, dict(horizon=24, vol_window=12)),
        (FixedHorizonLabeler, dict(horizon=24)),
        (DirectionalMeanReversionLabeler, dict(horizon=24, z_window=24)),
        (VolatilityRegimeLabeler, dict(horizon=24, lookback_window=120)),
        (MeanReversionEventLabeler, dict(horizon=24, z_window=24)),
        (MetaLabelLabeler, dict(horizon=24, mode="fixed_horizon")),
    ]


@pytest.mark.parametrize("cls,kwargs", _labeler_cases())
def test_labeler_labels_contract(small_ds, cls, kwargs):
    lab = cls(**kwargs)
    out = lab.labels(small_ds)

    assert out.columns == ["pair", "ts", "label"]
    assert out.height > 0
    assert out.schema["label"].is_numeric()

    assert out.get_column("label").drop_nulls().len() > 0

    assert isinstance(lab.horizon, int) and lab.horizon > 0


def test_labels_restrict_to_index(small_ds):
    from signalflow.target import FixedHorizonLabeler

    lab = FixedHorizonLabeler(horizon=24)
    at = small_ds.index().head(10)
    out = lab.labels(small_ds, at=at)
    assert out.height == 10
    assert out.select(["pair", "ts"]).equals(at.select(["pair", "ts"]))


def test_make_target_returns_working_labeler(small_ds):
    from signalflow.target import TripleBarrierLabeler, make_target

    clean = make_target("triple_barrier", max_bars=24)
    assert clean.labels(small_ds).columns == ["pair", "ts", "label"]

    legacy = make_target("triple_barrier_labeler", horizon=24, vol_window=12)
    assert isinstance(legacy, TripleBarrierLabeler)
    out = legacy.labels(small_ds)
    assert out.columns == ["pair", "ts", "label"]
    assert out.height > 0


@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_forecast_model_with_legacy_target(small_ds):
    from signalflow.target import TripleBarrierLabeler

    model = sf.ForecastModel(
        target=TripleBarrierLabeler(horizon=24, vol_window=12),
        features=sf.FeaturePipe(sf.SMA(20)),
        n_folds=3,
    )
    model.fit(small_ds)
    assert model.is_fitted


def _price_ds(closes):
    import datetime

    n = len(closes)
    ts = [datetime.datetime(2023, 1, 1) + datetime.timedelta(hours=i) for i in range(n)]
    frame = pl.DataFrame(
        {
            "pair": ["BTCUSDT"] * n,
            "ts": ts,
            "close": [float(c) for c in closes],
        }
    )
    return sf.Dataset(frame=frame)


def test_trend_break_yields_two_classes():
    import numpy as np

    from signalflow.target import TrendBreakLabeler

    seg = np.linspace(100.0, 200.0, 80)
    closes = np.concatenate([seg, seg[::-1], seg, seg[::-1], seg])
    ds = _price_ds(closes)

    lab = TrendBreakLabeler(window=20)
    out = lab.labels(ds)

    values = sorted(out.get_column("label").drop_nulls().unique().to_list())
    assert values == [0.0, 1.0]


def test_degenerate_target_raises_from_labels():
    import numpy as np

    from signalflow.target import TrendBreakLabeler

    closes = np.linspace(100.0, 500.0, 400)
    ds = _price_ds(closes)

    lab = TrendBreakLabeler(window=20)
    with pytest.raises(sf.DegenerateTargetError):
        lab.labels(ds)


def test_forecast_final_fit_degeneracy_raises_inner_fold_warns():
    from signalflow.target.base import LABEL_COL

    train = pl.DataFrame({LABEL_COL: [1.0] * 20})
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipe(),
        encode=None,
        select=None,
    )

    with pytest.raises(sf.DegenerateTargetError):
        model._fit_stack(train, fold=None)

    class _Fold:
        train_start_ts = None
        test_start_ts = None

    _enc, _sel, est = model._fit_stack(train, fold=_Fold())
    assert hasattr(est, "_sf_degenerate")


def test_fixed_horizon_duration_string_matches_bars():
    ds_1m = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-01-03", interval="1m")
    from signalflow.target import FixedHorizon

    by_duration = FixedHorizon(bars="1h").labels(ds_1m)
    by_bars = FixedHorizon(bars=60).labels(ds_1m)
    assert by_duration.equals(by_bars)


def test_triple_barrier_duration_string_resolves_against_interval(small_ds):
    from signalflow.target import TripleBarrierLabeler

    by_duration = TripleBarrierLabeler(horizon="1d", vol_window=12).labels(small_ds)
    by_bars = TripleBarrierLabeler(horizon=24, vol_window=12).labels(small_ds)
    assert by_duration.equals(by_bars)


def test_int_horizon_behaviour_unchanged(small_ds):
    from signalflow.target import TripleBarrierLabeler

    first = TripleBarrierLabeler(horizon=24, vol_window=12).labels(small_ds)
    second = TripleBarrierLabeler(horizon=24, vol_window=12).labels(small_ds)
    assert first.equals(second)


def test_window_duration_string_matches_bars(small_ds):
    from signalflow.target import VolatilityRegimeLabeler

    by_duration = VolatilityRegimeLabeler(horizon=24, lookback_window="5d").labels(small_ds)
    by_bars = VolatilityRegimeLabeler(horizon=24, lookback_window=120).labels(small_ds)
    assert by_duration.equals(by_bars)


def test_label_to_numeric_helper():
    from signalflow.target import Labeler

    s = pl.Series(["rise", "fall", "none", None, "mean_reverted"])
    num = Labeler._label_to_numeric(s)
    assert num.to_list() == [1.0, 0.0, 0.0, None, 1.0]

    n = pl.Series([0, 1, 2, 3])
    assert Labeler._label_to_numeric(n).to_list() == [0.0, 1.0, 2.0, 3.0]


def _frame(close, pair="BTCUSDT"):
    from datetime import datetime, timedelta

    n = len(close)
    ts = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)]
    c = list(map(float, close))
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "ts": ts,
            "open": c,
            "high": [x * 1.001 for x in c],
            "low": [x * 0.999 for x in c],
            "close": c,
            "volume": [1.0] * n,
        }
    )


def _ds(close):
    from signalflow.data.dataset import Dataset

    return Dataset(frame=_frame(close))


def test_vol_triple_barrier_scan():
    import numpy as np

    from signalflow.target import VolTripleBarrier

    n = 12
    sigma = np.full(n, 0.01)
    close = np.full(n, 100.0)
    flat_high = np.full(n, 100.0)
    flat_low = np.full(n, 100.0)
    t = VolTripleBarrier(tp_mult=2, sl_mult=1, max_bars=10)

    tp_high = flat_high.copy()
    tp_high[3] = 102.5
    assert t._scan(close, tp_high, flat_low, sigma)[0] == 1.0

    sl_low = flat_low.copy()
    sl_low[2] = 98.5
    assert t._scan(close, flat_high, sl_low, sigma)[0] == 0.0

    assert t._scan(close, flat_high, flat_low, sigma)[0] == 0.0
    t3 = VolTripleBarrier(tp_mult=2, sl_mult=1, max_bars=10, three_class=True)
    assert t3._scan(close, flat_high, flat_low, sigma)[0] == 2.0

    sig_nan = sigma.copy()
    sig_nan[0] = np.nan
    assert np.isnan(t._scan(close, tp_high, flat_low, sig_nan)[0])


def test_vol_horizon_labels():
    from signalflow.target import VolHorizon

    up = VolHorizon(bars=3, k=0.5, vol_window=50).labels(_ds([100.0] * 10 + [110.0] * 10))
    s = up.sort("ts").get_column("label").to_list()
    assert s[7] == 1
    assert s[2] is None

    down = VolHorizon(bars=3, k=0.5, vol_window=50).labels(_ds([110.0] * 10 + [100.0] * 10))
    assert down.sort("ts").get_column("label").to_list()[7] == 0


def test_reversion_barrier_scan():
    import numpy as np

    from signalflow.target import ReversionBarrier

    n = 10
    close = np.full(n, 98.0)
    anchor = np.full(n, 100.0)
    flat_high = np.full(n, 98.0)
    flat_low = np.full(n, 98.0)
    r = ReversionBarrier(sl_pct=0.02, max_bars=5)

    recross = flat_high.copy()
    recross[2] = 100.5
    assert r._scan(close, recross, flat_low, anchor)[0] == 1.0

    stopped = flat_low.copy()
    stopped[1] = 95.0
    assert r._scan(close, flat_high, stopped, anchor)[0] == 0.0

    above = np.full(n, 101.0)
    assert np.isnan(r._scan(above, recross, flat_low, anchor)[0])


def test_new_targets_registered_and_fit():
    from signalflow.enums import ComponentType
    from signalflow.target.base import make_target

    for name in ("vol_triple_barrier", "vol_horizon", "reversion_barrier"):
        cls = sf.registry.get(ComponentType.TARGET, name)
        assert sf.registry.get_info(ComponentType.TARGET, name).legacy is False
        cfg = cls().to_config()
        assert type(make_target(cfg["target"], **cfg["params"])) is cls

    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-05-01", interval="1h")
    model = sf.ForecastModel(
        target=sf.VolHorizon(bars=12), features=sf.FeaturePipe(sf.SMA(10)), encode=None, select=None
    ).fit(ds)
    assert model.is_fitted
    for scan_target in (sf.VolTripleBarrier(max_bars=12), sf.ReversionBarrier(anchor_bars=48, max_bars=24)):
        assert scan_target.labels(ds).columns == ["pair", "ts", "label"]
