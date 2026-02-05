import optuna
from typing import Any


def build_optuna_params(trial: optuna.Trial, tune_space: dict[str, tuple]) -> dict[str, Any]:
    """Build hyperparameters from optuna trial."""
    params = {}
    for name, spec in tune_space.items():
        param_type = spec[0]
        if param_type == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif param_type == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif param_type == "log_float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params
