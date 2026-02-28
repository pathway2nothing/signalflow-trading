"""Tests for signalflow.validator.base.SignalValidator."""

import polars as pl
import pytest

from signalflow.core import Signals
from signalflow.validator.base import SignalValidator


class TestSignalValidatorBase:
    """Tests for SignalValidator base class."""

    def test_default_attributes(self):
        """Test default attribute values."""
        v = SignalValidator()
        assert v.model is None
        assert v.model_type is None
        assert v.model_params is None
        assert v.train_params is None
        assert v.tune_enabled is False
        assert v.tune_params is None
        assert v.feature_columns is None
        assert v.pair_col == "pair"
        assert v.ts_col == "timestamp"

    def test_fit_not_implemented(self):
        """Test fit raises NotImplementedError."""
        v = SignalValidator()
        X = pl.DataFrame({"a": [1, 2, 3]})
        y = pl.Series([0, 1, 0])
        with pytest.raises(NotImplementedError, match="Subclasses must implement fit"):
            v.fit(X, y)

    def test_tune_disabled_raises(self):
        """Test tune raises ValueError when not enabled."""
        v = SignalValidator(tune_enabled=False)
        X = pl.DataFrame({"a": [1, 2, 3]})
        y = pl.Series([0, 1, 0])
        with pytest.raises(ValueError, match="Tuning is not enabled"):
            v.tune(X, y)

    def test_tune_enabled_not_implemented(self):
        """Test tune raises NotImplementedError when enabled but not overridden."""
        v = SignalValidator(tune_enabled=True)
        X = pl.DataFrame({"a": [1, 2, 3]})
        y = pl.Series([0, 1, 0])
        with pytest.raises(NotImplementedError, match="Subclasses must implement tune"):
            v.tune(X, y)

    def test_predict_not_implemented(self):
        """Test predict raises NotImplementedError."""
        v = SignalValidator()
        signals = Signals(pl.DataFrame({"pair": ["BTCUSDT"], "signal": [1]}))
        X = pl.DataFrame({"a": [1]})
        with pytest.raises(NotImplementedError, match="Subclasses must implement predict"):
            v.predict(signals, X)

    def test_predict_proba_not_implemented(self):
        """Test predict_proba raises NotImplementedError."""
        v = SignalValidator()
        signals = Signals(pl.DataFrame({"pair": ["BTCUSDT"], "signal": [1]}))
        X = pl.DataFrame({"a": [1]})
        with pytest.raises(NotImplementedError, match="Subclasses must implement predict_proba"):
            v.predict_proba(signals, X)

    def test_validate_signals_not_implemented(self):
        """Test validate_signals raises NotImplementedError."""
        v = SignalValidator()
        signals = Signals(pl.DataFrame({"pair": ["BTCUSDT"], "signal": [1]}))
        features = pl.DataFrame({"a": [1]})
        with pytest.raises(NotImplementedError, match="Subclasses must implement validate_signals"):
            v.validate_signals(signals, features)

    def test_save_not_implemented(self):
        """Test save raises NotImplementedError."""
        v = SignalValidator()
        with pytest.raises(NotImplementedError, match="Subclasses must implement save"):
            v.save("/tmp/model.pkl")

    def test_load_not_implemented(self):
        """Test load raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Subclasses must implement load"):
            SignalValidator.load("/tmp/model.pkl")

    def test_component_type(self):
        """Test component_type is set correctly."""
        from signalflow.core import SfComponentType

        assert SignalValidator.component_type == SfComponentType.VALIDATOR

    def test_custom_params(self):
        """Test validator with custom parameters."""
        v = SignalValidator(
            model_type="lightgbm",
            model_params={"n_estimators": 100},
            train_params={"early_stopping_rounds": 10},
            tune_enabled=True,
            tune_params={"n_trials": 50},
            pair_col="symbol",
            ts_col="time",
        )
        assert v.model_type == "lightgbm"
        assert v.model_params == {"n_estimators": 100}
        assert v.train_params == {"early_stopping_rounds": 10}
        assert v.tune_enabled is True
        assert v.tune_params == {"n_trials": 50}
        assert v.pair_col == "symbol"
        assert v.ts_col == "time"
