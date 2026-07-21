"""Walk-forward orchestration tests."""

import warnings

import pytest

from signalflow.data import data
from signalflow.model import ForecastModel, WalkForwardResult, walk_forward
from signalflow.target import FixedHorizon
from signalflow.transform import SMA, FeaturePipe

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")


@pytest.fixture(scope="module")
def wf_result():
    ds = data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-05-01", interval="1h")
    model = ForecastModel(
        backend="lightgbm",
        target=FixedHorizon(bars=12),
        features=FeaturePipe(SMA(20), SMA(10)),
        n_folds=3,
    )
    return walk_forward(model, ds, train="30d", step="30d"), ds


def test_three_non_overlapping_folds(wf_result):
    result, _ = wf_result
    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) == 3
    for a, b in zip(result.folds[:-1], result.folds[1:], strict=True):
        assert a.train_start < a.train_end == a.test_start < a.test_end
        assert a.test_end == b.test_start


def test_fold_oos_inside_its_test_window(wf_result):
    result, _ = wf_result
    for fold in result.folds:
        ts = fold.oos.get_column("ts")
        assert ts.min() >= fold.test_start
        assert ts.max() < fold.test_end
        assert set(fold.oos.columns) >= {"pair", "ts", "p_rise", "label"}


def test_merged_oos_has_no_duplicates(wf_result):
    result, _ = wf_result
    merged = result.oos()
    keyed = merged.select(["pair", "ts"])
    assert keyed.n_unique() == keyed.height


def test_evaluate_one_row_per_fold(wf_result):
    result, _ = wf_result
    scores = result.evaluate(lambda df: float(df.get_column("p_rise").mean()))
    assert scores.height == len(result.folds)
    assert "score" in scores.columns
    assert scores.get_column("score").null_count() == 0


def test_fold_predictions_match_full_history_predictions():
    ds = data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-06-01", interval="1h")
    model = ForecastModel(
        target=FixedHorizon(bars=12),
        features=FeaturePipe(SMA(50)),
        encode=None,
        select=None,
    )
    result = walk_forward(model, ds, train="60d", step="10d")
    assert result.folds
    for fold in result.folds:
        full = fold.model.predict(ds).rename({"p_rise": "p_full"})
        joined = fold.oos.join(full, on=["pair", "ts"], how="inner")
        assert joined.height == fold.oos.height
        diff = (joined.get_column("p_rise") - joined.get_column("p_full")).abs().max()
        assert diff == 0.0
