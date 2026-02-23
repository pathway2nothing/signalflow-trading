"""Tests for SfTorchModuleMixin in base_mixin.py."""

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
    def search_space(cls, model_size: str = "small") -> dict:
        size_config = {
            "small": {"hidden": (32, 64), "layers": (1, 2)},
            "medium": {"hidden": (64, 128), "layers": (2, 3)},
            "large": {"hidden": (128, 256), "layers": (3, 5)},
        }
        config = size_config[model_size]
        return {
            "input_size": 10,
            "hidden_size": {"type": "int", "low": config["hidden"][0], "high": config["hidden"][1]},
            "num_layers": {"type": "int", "low": config["layers"][0], "high": config["layers"][1]},
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

    def test_search_space_small(self):
        """Test search_space with small model size."""
        space = MockTorchModule.search_space("small")
        assert space["input_size"] == 10
        assert space["hidden_size"] == {"type": "int", "low": 32, "high": 64}
        assert space["num_layers"] == {"type": "int", "low": 1, "high": 2}

    def test_search_space_medium(self):
        """Test search_space with medium model size."""
        space = MockTorchModule.search_space("medium")
        assert space["input_size"] == 10
        assert space["hidden_size"] == {"type": "int", "low": 64, "high": 128}
        assert space["num_layers"] == {"type": "int", "low": 2, "high": 3}

    def test_search_space_large(self):
        """Test search_space with large model size."""
        space = MockTorchModule.search_space("large")
        assert space["input_size"] == 10
        assert space["hidden_size"] == {"type": "int", "low": 128, "high": 256}
        assert space["num_layers"] == {"type": "int", "low": 3, "high": 5}

    def test_search_space_has_type_keys(self):
        """Test that tunable params have proper type specification."""
        space = MockTorchModule.search_space("small")
        for key, val in space.items():
            if isinstance(val, dict):
                assert "type" in val, f"Param '{key}' missing 'type' key"
                assert val["type"] in ("int", "float", "categorical")

    def test_search_space_default_size(self):
        """Test that search_space defaults to small."""
        default_space = MockTorchModule.search_space()
        small_space = MockTorchModule.search_space("small")
        assert default_space == small_space
