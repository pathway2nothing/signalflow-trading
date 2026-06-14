"""Tests for the restored legacy labelers, adapted to the V5 Target API."""


import warnings

import polars as pl
import pytest

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import signalflow as sf


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


def test_label_to_numeric_helper():
    from signalflow.target import Labeler

    s = pl.Series(["rise", "fall", "none", None, "mean_reverted"])
    num = Labeler._label_to_numeric(s)
    assert num.to_list() == [1.0, 0.0, 0.0, None, 1.0]


    n = pl.Series([0, 1, 2, 3])
    assert Labeler._label_to_numeric(n).to_list() == [0.0, 1.0, 2.0, 3.0]
