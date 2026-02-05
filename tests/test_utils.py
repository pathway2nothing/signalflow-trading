"""Tests for signalflow.utils.import_utils and tune_utils."""

from unittest.mock import MagicMock

import pytest

from signalflow.utils.import_utils import import_model_class


class TestImportModelClass:
    def test_valid_import(self):
        cls = import_model_class("signalflow.core.enums.SignalType")
        from signalflow.core.enums import SignalType

        assert cls is SignalType

    def test_returns_type(self):
        cls = import_model_class("signalflow.core.enums.PositionType")
        assert isinstance(cls, type)

    def test_no_dot_raises(self):
        with pytest.raises(ValueError, match="Invalid class path"):
            import_model_class("NoDotHere")

    def test_invalid_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            import_model_class("nonexistent_module_xyz.SomeClass")

    def test_invalid_class_raises(self):
        with pytest.raises(AttributeError):
            import_model_class("signalflow.core.enums.DoesNotExist")

    def test_deeply_nested_path(self):
        cls = import_model_class("signalflow.core.containers.position.Position")
        from signalflow.core.containers.position import Position

        assert cls is Position


class TestBuildOptunaParams:
    @pytest.fixture
    def mock_trial(self):
        trial = MagicMock()
        trial.suggest_int.return_value = 10
        trial.suggest_float.return_value = 0.5
        trial.suggest_categorical.return_value = "a"
        return trial

    def test_int_param(self, mock_trial):
        from signalflow.utils.tune_utils import build_optuna_params

        result = build_optuna_params(mock_trial, {"n": ("int", 1, 100)})
        mock_trial.suggest_int.assert_called_once_with("n", 1, 100)
        assert result == {"n": 10}

    def test_float_param(self, mock_trial):
        from signalflow.utils.tune_utils import build_optuna_params

        result = build_optuna_params(mock_trial, {"lr": ("float", 0.0, 1.0)})
        mock_trial.suggest_float.assert_called_once_with("lr", 0.0, 1.0)
        assert result == {"lr": 0.5}

    def test_log_float_param(self, mock_trial):
        from signalflow.utils.tune_utils import build_optuna_params

        result = build_optuna_params(mock_trial, {"lr": ("log_float", 0.001, 1.0)})
        mock_trial.suggest_float.assert_called_once_with("lr", 0.001, 1.0, log=True)
        assert result == {"lr": 0.5}

    def test_categorical_param(self, mock_trial):
        from signalflow.utils.tune_utils import build_optuna_params

        result = build_optuna_params(mock_trial, {"act": ("categorical", ["a", "b"])})
        mock_trial.suggest_categorical.assert_called_once_with("act", ["a", "b"])
        assert result == {"act": "a"}
