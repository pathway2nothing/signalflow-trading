"""Tests for sklearn-based signal validators."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core import Signals
from signalflow.validator import (
    RandomForestValidator,
    LogisticRegressionValidator,
    SVMValidator,
    LightGBMValidator,
    XGBoostValidator,
    AutoSelectValidator,
    SklearnSignalValidator,
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


# ── RandomForestValidator ─────────────────────────────────────────────────


class TestRandomForestValidator:
    def test_create_model(self):
        v = RandomForestValidator()
        model = v._create_model()
        assert hasattr(model, "fit")

    def test_create_with_custom_params(self):
        v = RandomForestValidator(n_estimators=10)
        model = v._create_model()
        assert model.n_estimators == 10

    def test_fit(self):
        v = RandomForestValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model is not None

    def test_predict(self):
        v = RandomForestValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.predict(signals, X_test)
        assert "validation_pred" in result.value.columns
        assert result.value.height == 20

    def test_predict_proba(self):
        v = RandomForestValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.predict_proba(signals, X_test)
        prob_cols = [c for c in result.value.columns if c.startswith("probability_")]
        assert len(prob_cols) > 0

    def test_validate_signals(self):
        v = RandomForestValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.validate_signals(signals, X_test)
        prob_cols = [c for c in result.value.columns if c.startswith("probability_")]
        assert len(prob_cols) > 0


# ── LogisticRegressionValidator ───────────────────────────────────────────


class TestLogisticRegressionValidator:
    def test_create_model(self):
        v = LogisticRegressionValidator()
        model = v._create_model()
        assert hasattr(model, "fit")

    def test_fit(self):
        v = LogisticRegressionValidator()
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model is not None

    def test_custom_C(self):
        v = LogisticRegressionValidator(C=0.5)
        model = v._create_model()
        assert model.C == 0.5


# ── SVMValidator ──────────────────────────────────────────────────────────


class TestSVMValidator:
    def test_create_model(self):
        v = SVMValidator()
        model = v._create_model()
        assert hasattr(model, "fit")
        assert model.probability is True  # Required for predict_proba

    def test_custom_kernel(self):
        v = SVMValidator(kernel="linear")
        model = v._create_model()
        assert model.kernel == "linear"


# ── Feature extraction ───────────────────────────────────────────────────


class TestFeatureExtraction:
    def test_extract_features_fit_mode(self):
        v = RandomForestValidator()
        X, _ = _make_data(n=20, n_features=3)
        arr = v._extract_features(X, fit_mode=True)
        assert arr.shape == (20, 3)
        assert v.feature_columns == ["feat_0", "feat_1", "feat_2"]

    def test_extract_features_no_fit_raises(self):
        v = RandomForestValidator()
        X, _ = _make_data(n=20)
        with pytest.raises(ValueError, match="feature_columns not set"):
            v._extract_features(X)

    def test_extract_features_missing_col_raises(self):
        v = RandomForestValidator()
        v.feature_columns = ["nonexistent"]
        X, _ = _make_data(n=20)
        with pytest.raises(ValueError, match="Missing feature columns"):
            v._extract_features(X)

    def test_extract_labels_dataframe(self):
        v = RandomForestValidator()
        _, y = _make_data(n=20)
        arr = v._extract_labels(y)
        assert arr.shape == (20,)

    def test_extract_labels_series(self):
        v = RandomForestValidator()
        _, y = _make_data(n=20)
        arr = v._extract_labels(y["label"])
        assert arr.shape == (20,)

    def test_extract_labels_multi_col_with_label(self):
        v = RandomForestValidator()
        df = pl.DataFrame({"label": [0, 1, 0], "extra": [1.0, 2.0, 3.0]})
        arr = v._extract_labels(df)
        assert arr.shape == (3,)

    def test_extract_labels_multi_col_no_label_raises(self):
        v = RandomForestValidator()
        df = pl.DataFrame({"a": [0, 1], "b": [2, 3]})
        with pytest.raises(ValueError, match="label"):
            v._extract_labels(df)


# ── Fit and predict errors ────────────────────────────────────────────────


class TestFitPredictErrors:
    def test_predict_not_fitted_raises(self):
        v = RandomForestValidator()
        signals, X_test = _make_signals(n=10)
        with pytest.raises(ValueError, match="not fitted"):
            v.predict(signals, X_test)

    def test_predict_proba_not_fitted_raises(self):
        v = RandomForestValidator()
        signals, X_test = _make_signals(n=10)
        with pytest.raises(ValueError, match="not fitted"):
            v.predict_proba(signals, X_test)

    def test_get_class_labels(self):
        v = RandomForestValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        labels = v._get_class_labels()
        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_get_class_labels_not_fitted(self):
        v = RandomForestValidator()
        with pytest.raises(ValueError, match="not fitted"):
            v._get_class_labels()


# ── Save/load ────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        v = RandomForestValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)

        path = tmp_path / "validator.pkl"
        v.save(path)

        loaded = RandomForestValidator.load(path)
        assert loaded.feature_columns == v.feature_columns

        signals, X_test = _make_signals(n=10)
        result = loaded.predict(signals, X_test)
        assert result.value.height == 10


# ── Post-init defaults ──────────────────────────────────────────────────


class TestPostInit:
    def test_defaults(self):
        v = RandomForestValidator()
        assert v.model_params is not None
        assert v.train_params == {}
        assert v.tune_params is not None
        assert "n_trials" in v.tune_params


# ── AutoSelectValidator ────────────────────────────────────────────────


class TestAutoSelect:
    def test_auto_select_model(self):
        v = AutoSelectValidator()
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.selected_validator is not None
        assert v.model is not None

    def test_auto_select_predict(self):
        v = AutoSelectValidator()
        X, y = _make_data(n=100)
        v.fit(X, y)
        signals, X_test = _make_signals(n=20)
        result = v.predict(signals, X_test)
        assert result.value.height == 20

    def test_sklearn_signal_validator_alias(self):
        """SklearnSignalValidator should be an alias for AutoSelectValidator."""
        assert SklearnSignalValidator is AutoSelectValidator


# ── Tune ────────────────────────────────────────────────────────────────


class TestTune:
    def test_tune_with_rf(self):
        v = RandomForestValidator()
        X, y = _make_data(n=100)
        v.tune_params = {"n_trials": 2, "cv_folds": 2, "timeout": 10}
        best_params = v.tune(X, y)
        assert "n_estimators" in best_params

    def test_tune_auto_raises(self):
        v = AutoSelectValidator()
        X, y = _make_data(n=100)
        with pytest.raises(NotImplementedError):
            v.tune(X, y)


# ── LightGBM (if available) ─────────────────────────────────────────────


class TestLightGBMValidator:
    @pytest.fixture(autouse=True)
    def skip_if_no_lightgbm(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            pytest.skip("lightgbm not installed")

    def test_create_model(self):
        v = LightGBMValidator()
        model = v._create_model()
        assert hasattr(model, "fit")

    def test_fit(self):
        v = LightGBMValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model is not None

    def test_custom_params(self):
        v = LightGBMValidator(n_estimators=50, learning_rate=0.05)
        assert v.model_params["n_estimators"] == 50
        assert v.model_params["learning_rate"] == 0.05


# ── XGBoost (if available) ──────────────────────────────────────────────


class TestXGBoostValidator:
    @pytest.fixture(autouse=True)
    def skip_if_no_xgboost(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost not installed")

    def test_create_model(self):
        v = XGBoostValidator()
        model = v._create_model()
        assert hasattr(model, "fit")

    def test_fit(self):
        v = XGBoostValidator(n_estimators=10)
        X, y = _make_data(n=100)
        v.fit(X, y)
        assert v.model is not None

    def test_custom_params(self):
        v = XGBoostValidator(n_estimators=50, learning_rate=0.05)
        assert v.model_params["n_estimators"] == 50
        assert v.model_params["learning_rate"] == 0.05
