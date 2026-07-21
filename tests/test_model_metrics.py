"""Classification-quality metrics on OOS predictions and walk-forward presets."""

import warnings

import pytest

import signalflow as sf

warnings.filterwarnings("ignore", message="X does not have valid feature names")

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names")


def _model_and_ds():
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-03-01", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipe(sf.SMA(20), sf.SMA(50)),
        encode=None,
        select=None,
    ).fit(ds)
    return model, ds


def test_classification_scorecard_keys_and_ranges():
    model, ds = _model_and_ds()
    card = sf.classification_scorecard(model, ds)
    for key in ("n", "threshold", "base_rate", "auc", "pr_auc", "brier", "precision", "recall", "f1"):
        assert key in card
    assert card["n"] > 0
    assert 0.0 <= card["auc"] <= 1.0


def test_classification_scorecard_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    from signalflow.model.metrics import _joined

    model, ds = _model_and_ds()
    card = sf.classification_scorecard(model, ds)
    df = _joined(model, ds)
    y = df.get_column("label").cast(int).to_numpy()
    p = df.get_column(model.output).to_numpy()
    assert card["auc"] == pytest.approx(float(roc_auc_score(y, p)))


def test_walk_forward_evaluate_auc_preset():
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-06-01", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12), features=sf.FeaturePipe(sf.SMA(20)), encode=None, select=None
    )
    result = sf.walk_forward(model, ds, train="90d", step="30d")
    scores = result.evaluate("auc")
    assert scores.height == len(result.folds)
    finite = scores.get_column("score").drop_nulls().to_numpy()
    assert all((v != v) or (0.0 <= v <= 1.0) for v in finite)


def test_walk_forward_evaluate_unknown_preset_raises():
    ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-04-01", interval="1h")
    model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=12), features=sf.FeaturePipe(sf.SMA(20)), encode=None, select=None
    )
    result = sf.walk_forward(model, ds, train="60d", step="30d")
    with pytest.raises(ValueError, match="unknown metric preset"):
        result.evaluate("nope")
