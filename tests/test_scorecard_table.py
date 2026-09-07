"""One scorecard row per model or fold, with train-calibrated operating points and per-target means."""

import math

import polars as pl
import pytest

import signalflow as sf
from signalflow.model.metrics import METRICS, resolve_operating


def test_resolve_operating_forms():
    scores = pl.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    train = pl.Series([0.9, 0.9, 0.9])
    assert resolve_operating(0.42, None, scores) == 0.42
    assert resolve_operating("q0.5", train, scores) == pytest.approx(0.3)
    assert resolve_operating("train_q0.5", train, scores) == pytest.approx(0.9)
    assert resolve_operating("train_q0.5", None, scores) == pytest.approx(0.3)  # falls back to the OOS scores
    with pytest.raises(ValueError):
        resolve_operating("bogus", None, scores)
    with pytest.raises(ValueError):
        resolve_operating("q1.5", None, scores)


def test_table_per_model_and_means(ds, fitted_forecast):
    table = sf.scorecard_table({"rise12": fitted_forecast, "again": fitted_forecast}, ds, operating="q0.8")
    assert table.height == 2
    assert table.get_column("model").to_list() == ["rise12", "again"]
    for col in ("target", "target_params", "n_test", "prevalence", "threshold", *METRICS):
        assert col in table.columns
    assert table.get_column("n_test")[0] > 0
    assert 0.0 < table.get_column("prevalence")[0] < 1.0
    assert 0.0 <= table.get_column("roc_auc")[0] <= 1.0
    means = sf.scorecard_means(table)
    assert means.height == 1 and means.get_column("rows")[0] == 2
    assert means.get_column("roc_auc")[0] == pytest.approx(table.get_column("roc_auc").mean())

    single = sf.scorecard_table(fitted_forecast, ds, operating=0.5, metrics=("auc", "f1"))
    assert single.height == 1 and {"auc", "f1"} <= set(single.columns) and "brier" not in single.columns
    with pytest.raises(ValueError):
        sf.scorecard_table(fitted_forecast, ds, metrics=("nope",))


def test_table_per_fold_uses_the_training_window_threshold():
    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2023-01-01", end="2023-05-01", interval="1h")
    model = sf.ForecastModel(
        backend="lightgbm",
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipeline(sf.SMA(20), sf.SMA(10)),
        cv=sf.KFold(3),
    )
    result = sf.walk_forward(model, ds, train="30d", step="30d")
    table = sf.scorecard_table(result, ds, operating="train_q0.9")
    assert table.height == len(result.folds)
    assert table.get_column("fold").to_list() == list(range(len(result.folds)))
    assert table.get_column("tag").to_list() == [f.tag for f in result.folds]
    assert all(len(t) == 6 for t in table.get_column("tag").to_list())
    assert (table.get_column("n_test") > 0).all()
    assert all(not math.isnan(v) for v in table.get_column("threshold").to_list())
    # a train-calibrated threshold differs from the OOS quantile in general
    oos_table = sf.scorecard_table(result, ds, operating="q0.9")
    assert oos_table.get_column("threshold").to_list() != table.get_column("threshold").to_list()
    means = sf.scorecard_means(table, by="target")
    assert means.height == 1 and means.get_column("rows")[0] == len(result.folds)


def test_classification_scorecard_accepts_operating_specs(ds, fitted_forecast):
    fixed = sf.classification_scorecard(fitted_forecast, ds, threshold=0.5)
    calibrated = sf.classification_scorecard(fitted_forecast, ds, threshold="train_q0.9")
    assert fixed["threshold"] == 0.5 and 0.0 < calibrated["threshold"] < 1.0
    assert set(fixed) == set(calibrated)
