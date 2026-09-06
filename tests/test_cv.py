"""CV schemes: KFold / Rolling fold layout, expanding window, config round-trip, short-span fallback."""

from datetime import datetime, timedelta

import signalflow as sf
from signalflow.model.cv import build_cv


def _hours(n: int) -> list[datetime]:
    return [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]


def test_kfold_blocks_are_contiguous_and_expanding():
    ts = _hours(24 * 10)
    folds = sf.KFold(4).folds(ts, timedelta(hours=6))
    assert len(folds) == 3
    assert all(f.train_start is None for f in folds)
    assert folds[0].test_end < folds[1].test_start < folds[2].test_start


def test_rolling_trailing_window_and_expanding():
    ts = _hours(24 * 10)
    rolling = sf.Rolling(step="1d", window="3d").folds(ts, timedelta(hours=6))
    assert len(rolling) >= 8
    assert all(f.train_end - f.train_start == timedelta(days=3) for f in rolling)
    expanding = sf.Rolling(step="1d", window=None).folds(ts, timedelta(hours=6))
    assert all(f.train_start is None for f in expanding)
    assert [f.test_start for f in expanding] == [f.test_start for f in rolling]


def test_rolling_falls_back_to_kfold_on_short_span():
    ts = _hours(6)
    folds = sf.Rolling(step="30d").folds(ts, timedelta(hours=1))
    assert 1 <= len(folds) <= 2  # KFold(3) over 6 timestamps


def test_cv_config_round_trip():
    for scheme in (sf.KFold(5), sf.Rolling(step="7d", window="90d"), sf.Rolling(step="1d", window=None)):
        assert build_cv(scheme.to_config()) == scheme
    assert build_cv(None) == sf.Rolling()
    assert build_cv(sf.KFold(2)) == sf.KFold(2)


def test_model_cv_is_recorded_in_fingerprint():
    ds = sf.data("synthetic", pairs=["BTCUSDT"], start="2024-01-01", end="2024-02-01", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=6), features=sf.FeaturePipeline(sf.SMA(5)), cv={"scheme": "kfold", "n": 3}
    ).fit(ds)
    assert model.cv == sf.KFold(3)
    assert model.fingerprint["cv"]["scheme"] == "kfold"
    assert model.fingerprint["cv"]["n_folds_effective"] == 2
