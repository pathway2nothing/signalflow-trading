"""Tests for SklearnSignalValidator."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from signalflow.core import Signals
from signalflow.validator.sklearn_validator import (
    SklearnSignalValidator,
    SKLEARN_MODELS,
)


TS = datetime(2024, 1, 1)


def _make_data(n=200, n_features=5):
    """Create synthetic features + labels for classification."""
    np.random.seed(42)
    pairs = ["BTCUSDT"] * n
    timestamps = [TS + timedelta(minutes=i) for i in range(n)]
    features = np.random.randn(n, n_features)
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)  # 0 or 1

    cols = {"pair": pairs, "timestamp": timestamps}
    for i in range(n_features):
        cols[f"feat_{i}"] = features[:, i].tolist()

    X = pl.DataFrame(cols)
    y = pl.DataFrame({"label": labels})
    return X, y


def _make_signals(n=50, n_features=5):
    """Create signals and matching features for prediction."""
    np.random.seed(123)
    pairs = ["BTCUSDT"] * n
    timestamps = [TS + timedelta(minutes=i) for i in range(n)]

    sig_data = {
        "pair": pairs,
        "timestamp": timestamps,
        "signal": [1] * n,
        "signal_type": ["rise"] * n,
    }
    signals = Signals(pl.DataFrame(sig_data))

    feat_data = {"pair": pairs, "timestamp": timestamps}
    features = np.random.randn(n, n_features)
    for i in range(n_features):
        feat_data[f"feat_{i}"] = features[:, i].tolist()
    X = pl.DataFrame(feat_data)

    return signals, X


# ── Model config ─────────────────────────────────────────────────────────


class TestModelConfig:
    def test_known_models(self):
        assert "random_forest" in SKLEARN_MODELS
        assert "logistic_regression" in SKLEARN_MODELS
        assert "svm" in SKLEARN_MODELS

    def test_unknown_model_raises(self):
        v = SklearnSignalValidator(model_type="nonexistent")
        with pytest.raises(ValueError, match="Unknown model_type"):
            v._get_model_config("nonexistent")

    def test_create_random_forest(self):
        v = SklearnSignalValidator(model_type="random_forest")
        model = v._create_model("random_forest")
        assert hasattr(model, "fit")

    def test_create_logistic_regression(self):
        v = SklearnSignalValidator(model_type="logistic_regression")
        model = v._create_model("logistic_regression")
        assert hasattr(model, "fit")

    def test_create_with_custom_params(self):
        v = SklearnSignalValidator(model_type="random_forest")
        model = v._create_model("random_forest", {"n_estimators": 10})
        assert model.n_estimators == 10


# ── Feature extraction ───────────────────────────────────────────────────


class TestFeatureExtraction:
    def test_extract_features_fit_mode(self):
        v = SklearnSignalValidator(model_type="random_forest")
        X, _ = _make_data(n=20, n_features=3)
        arr = v._extract_features(X, fit_mode=True)
        assert arr.shape == (20, 3)
        assert v.feature_columns == ["feat_0", "feat_1", "feat_2"]

    def test_extract_features_no_fit_raises(self):
        v = SklearnSignalValidator(model_type="random_forest")
        X, _ = _make_data(n=20)
        with pytest.raises(ValueError, match="feature_columns not set"):
            v._extract_features(X)

    def test_extract_features_missing_col_raises(self):
        v = SklearnSignalValidator(model_type="random_forest")
        v.feature_columns = ["nonexistent"]
        X, _ = _make_data(n=20)
        with pytest.raises(ValueError, match="Missing feature columns"):
            v._extract_features(X)

    def test_extract_labels_dataframe(self):
        v = SklearnSignalValidator()
        _, y = _make_data(n=20)
        arr = v._extract_labels(y)
        assert arr.shape == (20,)

    def test_extract_labels_series(self):
        v = SklearnSignalValidator()
        _, y = _make_data(n=20)
        arr = v._extract_labels(y["label"])
        assert arr.shape == (20,)

    def test_extract_labels_multi_col_with_label(self):
        v = SklearnSignalValidator()
        df = pl.DataFrame({"label": [0, 1, 0], "extra": [1.0, 2.0, 3.0]})
        arr = v._extract_labels(df)
        assert arr.shape == (3,)

    def test_extract_labels_multi_col_no_label_raises(self):
        v = SklearnSignalValidator()
        df = pl.DataFrame({"a": [0, 1], "b": [2, 3]})
        with pytest.raises(ValueError, match="label"):
            v._extract_labels(df)


# ── Fit and predict ──────────────────────────────────────────────────────


class TestFitPredict:
    def test_fit_random_forest(self):
        v = SklearnSignalValidator(model_type="random_forest")
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model is not None

    def test_fit_logistic_regression(self):
        v = SklearnSignalValidator(model_type="logistic_regression")
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model is not None

    def test_predict(self):
        v = SklearnSignalValidator(model_type="random_forest", model_params={"n_estimators": 10, "random_state": 42})
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.predict(signals, X_test)
        assert "validation_pred" in result.value.columns
        assert result.value.height == 20

    def test_predict_proba(self):
        v = SklearnSignalValidator(model_type="random_forest", model_params={"n_estimators": 10, "random_state": 42})
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.predict_proba(signals, X_test)
        # Should have probability columns
        prob_cols = [c for c in result.value.columns if c.startswith("probability_")]
        assert len(prob_cols) > 0

    def test_predict_not_fitted_raises(self):
        v = SklearnSignalValidator(model_type="random_forest")
        signals, X_test = _make_signals(n=10)
        with pytest.raises(ValueError, match="not fitted"):
            v.predict(signals, X_test)

    def test_predict_proba_not_fitted_raises(self):
        v = SklearnSignalValidator(model_type="random_forest")
        signals, X_test = _make_signals(n=10)
        with pytest.raises(ValueError, match="not fitted"):
            v.predict_proba(signals, X_test)

    def test_validate_signals(self):
        v = SklearnSignalValidator(model_type="random_forest", model_params={"n_estimators": 10, "random_state": 42})
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.validate_signals(signals, X_test)
        prob_cols = [c for c in result.value.columns if c.startswith("probability_")]
        assert len(prob_cols) > 0

    def test_get_class_labels(self):
        v = SklearnSignalValidator(model_type="random_forest", model_params={"n_estimators": 10, "random_state": 42})
        X, y = _make_data(n=100)
        v.fit(X, y)
        labels = v._get_class_labels()
        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_get_class_labels_not_fitted(self):
        v = SklearnSignalValidator()
        with pytest.raises(ValueError, match="not fitted"):
            v._get_class_labels()


# ── Save/load ────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        v = SklearnSignalValidator(model_type="random_forest", model_params={"n_estimators": 10, "random_state": 42})
        X, y = _make_data(n=100)
        v.fit(X, y)

        path = tmp_path / "validator.pkl"
        v.save(path)

        loaded = SklearnSignalValidator.load(path)
        assert loaded.model_type == "random_forest"
        assert loaded.feature_columns == v.feature_columns

        signals, X_test = _make_signals(n=10)
        result = loaded.predict(signals, X_test)
        assert result.value.height == 10


# ── Post-init defaults ──────────────────────────────────────────────────


class TestPostInit:
    def test_defaults(self):
        v = SklearnSignalValidator()
        assert v.model_params == {}
        assert v.train_params == {}
        assert v.tune_params is not None
        assert "n_trials" in v.tune_params


# ── Auto-select and tune ────────────────────────────────────────────────


class TestAutoSelect:
    def test_auto_select_model(self):
        v = SklearnSignalValidator(model_type="auto")
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model_type in ["random_forest", "logistic_regression"]
        assert v.model is not None

    def test_auto_select_sets_params(self):
        v = SklearnSignalValidator(model_type="auto")
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model_params is not None


class TestTune:
    def test_tune_with_rf(self):
        v = SklearnSignalValidator(model_type="random_forest")
        X, y = _make_data(n=100)
        v.tune_params = {"n_trials": 2, "cv_folds": 2, "timeout": 10}
        best_params = v.tune(X, y)
        assert "n_estimators" in best_params

    def test_tune_auto_raises(self):
        v = SklearnSignalValidator(model_type="auto")
        X, y = _make_data(n=100)
        with pytest.raises(ValueError, match="Set model_type"):
            v.tune(X, y)
