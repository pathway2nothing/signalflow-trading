from abc import ABC, abstractmethod
import optuna
from signalflow.core.enums import SfComponentType
from typing import Literal


class SfTorchModuleMixin(ABC):
    """Mixin for all SignalFlow neural network modules.

    Provides standardized interface for PyTorch modules used in SignalFlow,
    including default parameters and hyperparameter tuning via Optuna.

    All neural network modules (detectors, validators, etc.) should inherit
    from this mixin to ensure consistent configuration and tuning interfaces.

    Key features:
        - Automatic component type registration
        - Standardized parameter interface
        - Built-in Optuna integration for hyperparameter tuning
        - Size-based architecture variants (small, medium, large)

    Attributes:
        component_type (SfComponentType): Always set to TORCH_MODULE for registry.

    Example:
        ```python
        import torch
        import torch.nn as nn
        import optuna
        from signalflow.core import SfTorchModuleMixin

        class MyLSTMDetector(nn.Module, SfTorchModuleMixin):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, 3)  # 3 classes

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

            @classmethod
            def default_params(cls) -> dict:
                return {
                    "input_size": 10,
                    "hidden_size": 64,
                    "num_layers": 2
                }

            @classmethod
            def tune(cls, trial: optuna.Trial, model_size: str = "small") -> dict:
                if model_size == "small":
                    hidden_range = (32, 64)
                    layers_range = (1, 2)
                elif model_size == "medium":
                    hidden_range = (64, 128)
                    layers_range = (2, 3)
                else:  # large
                    hidden_range = (128, 256)
                    layers_range = (3, 4)

                return {
                    "input_size": 10,
                    "hidden_size": trial.suggest_int("hidden_size", *hidden_range),
                    "num_layers": trial.suggest_int("num_layers", *layers_range)
                }

        # Use default params
        model = MyLSTMDetector(**MyLSTMDetector.default_params())

        # Hyperparameter tuning
        study = optuna.create_study()

        def objective(trial):
            params = MyLSTMDetector.tune(trial, model_size="medium")
            model = MyLSTMDetector(**params)
            # ... train and evaluate model ...
            return validation_loss

        study.optimize(objective, n_trials=100)
        ```

    Note:
        Classes inheriting this mixin must implement both abstract methods.
        The component_type is automatically set for registry integration.

    See Also:
        sf_component: Decorator for registering modules in the registry.
        SfComponentType: Enum of all component types including TORCH_MODULE.
    """

    component_type: SfComponentType = SfComponentType.TORCH_MODULE

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        """Get default parameters for module instantiation.

        Provides sensible defaults for quick prototyping and baseline comparisons.
        These parameters should work reasonably well out-of-the-box.

        Returns:
            dict: Dictionary of parameter names and default values.
                Keys match constructor argument names.

        Example:
            ```python
            class MyTransformer(nn.Module, SfTorchModuleMixin):
                def __init__(self, d_model: int, nhead: int, num_layers: int):
                    super().__init__()
                    # ... initialization ...

                @classmethod
                def default_params(cls) -> dict:
                    return {
                        "d_model": 128,
                        "nhead": 8,
                        "num_layers": 3
                    }

            # Instantiate with defaults
            model = MyTransformer(**MyTransformer.default_params())

            # Override specific params
            params = MyTransformer.default_params()
            params["d_model"] = 256
            model = MyTransformer(**params)
            ```

        Note:
            Should be comprehensive - include all constructor parameters.
            Consider computational constraints when setting defaults.
        """
        ...

    @classmethod
    @abstractmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Define Optuna hyperparameter search space.

        Provides size-based architecture variants for different computational budgets:
            - small: Fast training, limited capacity
            - medium: Balanced performance/speed
            - large: Maximum capacity, slower training

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.
            model_size (Literal["small", "medium", "large"]): Architecture size variant.
                Default: "small".

        Returns:
            dict: Dictionary of hyperparameters sampled from search space.
                Keys match constructor argument names.

        Example:
            ```python
            class MyRNN(nn.Module, SfTorchModuleMixin):
                def __init__(self, input_size: int, hidden_size: int,
                           num_layers: int, dropout: float):
                    super().__init__()
                    # ... initialization ...

                @classmethod
                def default_params(cls) -> dict:
                    return {
                        "input_size": 20,
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.1
                    }

                @classmethod
                def tune(cls, trial: optuna.Trial, model_size: str = "small") -> dict:
                    # Size-based ranges
                    size_config = {
                        "small": {
                            "hidden": (32, 64),
                            "layers": (1, 2)
                        },
                        "medium": {
                            "hidden": (64, 128),
                            "layers": (2, 3)
                        },
                        "large": {
                            "hidden": (128, 256),
                            "layers": (3, 5)
                        }
                    }

                    config = size_config[model_size]

                    return {
                        "input_size": 20,
                        "hidden_size": trial.suggest_int(
                            "hidden_size",
                            *config["hidden"]
                        ),
                        "num_layers": trial.suggest_int(
                            "num_layers",
                            *config["layers"]
                        ),
                        "dropout": trial.suggest_float("dropout", 0.0, 0.5)
                    }

            # Run hyperparameter search
            import optuna

            def objective(trial):
                params = MyRNN.tune(trial, model_size="medium")
                model = MyRNN(**params)

                # Train model
                trainer = Trainer(model, train_data, val_data)
                val_loss = trainer.train()

                return val_loss

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)

            # Get best params
            best_params = study.best_params
            best_model = MyRNN(**MyRNN.tune(study.best_trial, "medium"))
            ```

        Note:
            Search space should be tailored to model_size.
            Use trial.suggest_* methods (int, float, categorical).
            Return dict must be compatible with constructor.
            Consider pruning for early stopping of poor trials.
        """
        ...
