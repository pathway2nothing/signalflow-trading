"""Experiment tracking - a thin MLflow run wrapper that no-ops without mlflow installed."""

import contextlib

from loguru import logger


@contextlib.contextmanager
def experiment_run(
    name: str, params: "dict | None" = None, tags: "dict | None" = None, tracking_uri: "str | None" = None
):
    """Open an MLflow run, log ``params``/``tags``, yield the mlflow module (or None)."""
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "experiment_run: mlflow is not installed; tracking disabled (pip install signalflow-trading[live])"
        )
        yield None
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name)
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        yield mlflow
