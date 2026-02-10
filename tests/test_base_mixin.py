"""Tests for SfTorchModuleMixin in base_mixin.py."""

import pytest
import optuna

from signalflow.core.base_mixin import SfTorchModuleMixin
from signalflow.core.enums import SfComponentType


class MockTorchModule(SfTorchModuleMixin):
    """Mock implementation of SfTorchModuleMixin for testing."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    @classmethod
    def default_params(cls) -> dict:
        return {
            "input_size": 10,
            "hidden_size": 64,
            "num_layers": 2,
        }

    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: str = "small") -> dict:
        size_config = {
            "small": {"hidden": (32, 64), "layers": (1, 2)},
            "medium": {"hidden": (64, 128), "layers": (2, 3)},
            "large": {"hidden": (128, 256), "layers": (3, 5)},
        }
        config = size_config[model_size]
        return {
            "input_size": 10,
            "hidden_size": trial.suggest_int("hidden_size", *config["hidden"]),
            "num_layers": trial.suggest_int("num_layers", *config["layers"]),
        }


class TestSfTorchModuleMixin:
    """Tests for SfTorchModuleMixin."""

    def test_component_type(self):
        """Test that component_type is TORCH_MODULE."""
        assert MockTorchModule.component_type == SfComponentType.TORCH_MODULE

    def test_default_params(self):
        """Test default_params returns expected dictionary."""
        params = MockTorchModule.default_params()
        assert params == {
            "input_size": 10,
            "hidden_size": 64,
            "num_layers": 2,
        }

    def test_instantiation_with_default_params(self):
        """Test module can be instantiated with default params."""
        model = MockTorchModule(**MockTorchModule.default_params())
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.num_layers == 2

    def test_tune_small_model(self):
        """Test tune method with small model size."""
        study = optuna.create_study()
        trial = study.ask()
        params = MockTorchModule.tune(trial, model_size="small")

        assert params["input_size"] == 10
        assert 32 <= params["hidden_size"] <= 64
        assert 1 <= params["num_layers"] <= 2

    def test_tune_medium_model(self):
        """Test tune method with medium model size."""
        study = optuna.create_study()
        trial = study.ask()
        params = MockTorchModule.tune(trial, model_size="medium")

        assert params["input_size"] == 10
        assert 64 <= params["hidden_size"] <= 128
        assert 2 <= params["num_layers"] <= 3

    def test_tune_large_model(self):
        """Test tune method with large model size."""
        study = optuna.create_study()
        trial = study.ask()
        params = MockTorchModule.tune(trial, model_size="large")

        assert params["input_size"] == 10
        assert 128 <= params["hidden_size"] <= 256
        assert 3 <= params["num_layers"] <= 5

    def test_tune_creates_valid_model(self):
        """Test that tune params can create valid model instance."""
        study = optuna.create_study()
        trial = study.ask()
        params = MockTorchModule.tune(trial, model_size="medium")
        model = MockTorchModule(**params)

        assert model.input_size == 10
        assert 64 <= model.hidden_size <= 128

    def test_optuna_optimization(self):
        """Test integration with Optuna optimization loop."""

        def objective(trial):
            params = MockTorchModule.tune(trial, model_size="small")
            model = MockTorchModule(**params)
            # Simulate loss based on hidden_size
            return abs(model.hidden_size - 48)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5, show_progress_bar=False)

        assert len(study.trials) == 5
        assert study.best_value is not None
