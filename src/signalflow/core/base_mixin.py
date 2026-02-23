from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from signalflow.core.enums import SfComponentType


class SfTorchModuleMixin(ABC):
    """Mixin for all SignalFlow neural network modules.

    Provides standardized interface for PyTorch modules used in SignalFlow,
    including default parameters and hyperparameter search space definition.

    All neural network modules (detectors, validators, etc.) should inherit
    from this mixin to ensure consistent configuration and tuning interfaces.

    Key features:
        - Automatic component type registration
        - Standardized parameter interface
        - Framework-agnostic hyperparameter search space
        - Size-based architecture variants (small, medium, large)

    Attributes:
        component_type (SfComponentType): Always set to TORCH_MODULE for registry.

    Example:
        ```python
        import torch.nn as nn
        from signalflow.core import SfTorchModuleMixin

        class MyLSTMDetector(nn.Module, SfTorchModuleMixin):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, 3)

            @classmethod
            def default_params(cls) -> dict:
                return {
                    "input_size": 10,
                    "hidden_size": 64,
                    "num_layers": 2,
                }

            @classmethod
            def search_space(cls, model_size: str = "small") -> dict:
                if model_size == "small":
                    hidden_range = (32, 64)
                    layers_range = (1, 2)
                elif model_size == "medium":
                    hidden_range = (64, 128)
                    layers_range = (2, 3)
                else:
                    hidden_range = (128, 256)
                    layers_range = (3, 4)

                return {
                    "input_size": 10,
                    "hidden_size": {"type": "int", "low": hidden_range[0], "high": hidden_range[1]},
                    "num_layers": {"type": "int", "low": layers_range[0], "high": layers_range[1]},
                }

        # Use default params
        model = MyLSTMDetector(**MyLSTMDetector.default_params())

        # Get search space for tuning
        space = MyLSTMDetector.search_space("medium")
        # space = {
        #     "input_size": 10,
        #     "hidden_size": {"type": "int", "low": 64, "high": 128},
        #     "num_layers": {"type": "int", "low": 2, "high": 3},
        # }
        ```

    See Also:
        sf_component: Decorator for registering modules in the registry.
        SfComponentType: Enum of all component types including TORCH_MODULE.
    """

    component_type: SfComponentType = SfComponentType.TORCH_MODULE

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        """Get default parameters for module instantiation.

        Returns:
            dict: Parameter names mapped to default values.
                Keys match constructor argument names.
        """
        ...

    @classmethod
    @abstractmethod
    def search_space(
        cls, model_size: Literal["small", "medium", "large"] = "small"
    ) -> dict:
        """Return hyperparameter search space for tuning.

        Describes tunable parameters with their types and bounds.
        Fixed values are plain Python values. Tunable parameters are
        dicts with a ``type`` key and range specification.

        Supported param specs::

            Fixed value:      42 or "gelu"
            Int range:        {"type": "int", "low": 32, "high": 256}
            Float range:      {"type": "float", "low": 0.0, "high": 0.5}
            Float log-scale:  {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}
            Categorical:      {"type": "categorical", "choices": ["relu", "gelu"]}

        Args:
            model_size: Architecture size variant.
                ``"small"`` — fast training, limited capacity.
                ``"medium"`` — balanced.
                ``"large"`` — maximum capacity, slower.

        Returns:
            dict: Parameter names mapped to fixed values or param specs.
        """
        ...
