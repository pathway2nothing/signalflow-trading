from abc import ABC, abstractmethod
import optuna
from signalflow.core.enums import SfComponentType
from typing import Literal


class SfTorchModuleMixin(ABC):
    """Mixin for all SignalFlow neural network modules."""
    
    component_type: SfComponentType = SfComponentType.TORCH_MODULE
    
    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        ...
    
    @classmethod
    @abstractmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Optuna search space."""
        ...