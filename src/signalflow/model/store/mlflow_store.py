"""
MLflow backend (real mlflow, local file tracking store by default).

URI forms (``mlflow://`` already stripped to ``location``)::

models/<name>            register/load the model named <name>
    models/<name>@<ref>      load a specific registered version/stage <ref>
    <name>                   shorthand for models/<name>

On save we open a run tagged with the model name, write the artifact layout to a
temp dir, log it under the ``model`` artifact path, and (best-effort) register a
model version in the registry. On load we resolve the artifacts: first via the
registry (``models:/<name>/<ref>``), and otherwise via the most-recent run tagged
with the model name - this keeps loads working on the local file store, whose
registry can be flaky in recent mlflow releases.

If no tracking URI is configured we default to a local ``./mlruns`` file store so
this works without a running server.
"""

import os
import tempfile
from pathlib import Path

from signalflow.errors import ArtifactError
from signalflow.model.store._layout import read_layout, write_layout

_ARTIFACT_PATH = "model"
_EXPERIMENT = "signalflow_models"
_NAME_TAG = "sf_model_name"


def _ensure_tracking_uri() -> None:
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    current = mlflow.get_tracking_uri()

    if not current or current in ("", "file:", "file:."):
        mlflow.set_tracking_uri(Path("./mlruns").resolve().as_uri())


def _parse_location(location: str) -> tuple[str, str | None]:
    """Return ``(model_name, ref)`` where ref is a registry version/stage."""
    loc = location.strip().strip("/")
    if loc.startswith("models/"):
        loc = loc[len("models/") :]
    name, sep, ref = loc.partition("@")
    if not name:
        raise ArtifactError(f"could not parse model name from mlflow location {location!r}")
    return name, (ref if sep else None)


def save(model, location: str) -> str:
    """Persist ``model`` under ``location``; nests inside the caller's active run if any.

    On a successful registry write the returned URI is pinned to the new version
    (``mlflow://models/<name>@<version>``) so a later load is reproducible; if
    registration fails the URI stays unversioned and load resolves ``latest``.
    """
    import mlflow
    from loguru import logger

    name, _ = _parse_location(location)
    _ensure_tracking_uri()

    inside_active_run = mlflow.active_run() is not None
    version: str | None = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            write_layout(model, tmp)
            run_kwargs: dict = {"tags": {_NAME_TAG: name}}
            if inside_active_run:
                run_kwargs["nested"] = True
            else:
                mlflow.set_experiment(_EXPERIMENT)
            with mlflow.start_run(**run_kwargs) as run:
                mlflow.log_artifacts(tmp, artifact_path=_ARTIFACT_PATH)
                model_uri = f"runs:/{run.info.run_id}/{_ARTIFACT_PATH}"
                try:
                    registered = mlflow.register_model(model_uri, name)
                    version = getattr(registered, "version", None)
                except Exception as exc:
                    logger.warning(
                        f"mlflow: registration failed for {name!r}; URI stays unversioned "
                        f"and load will resolve 'latest': {exc}"
                    )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"mlflow save failed for {location!r}: {exc}") from exc

    if version is not None:
        return f"mlflow://models/{name}@{version}"
    return f"mlflow://models/{name}"


def _download_from_registry(name: str, ref: str | None) -> str | None:
    import mlflow

    version = ref if ref else "latest"
    try:
        return mlflow.artifacts.download_artifacts(f"models:/{name}/{version}")
    except Exception:
        return None


def _download_from_run(name: str) -> str | None:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    if not experiment_ids:
        return None
    runs = client.search_runs(
        experiment_ids,
        filter_string=f"tags.{_NAME_TAG} = '{name}'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    run_id = runs[0].info.run_id
    try:
        return mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{_ARTIFACT_PATH}")
    except Exception:
        return None


def load(location: str):
    from loguru import logger

    name, ref = _parse_location(location)
    _ensure_tracking_uri()
    if ref is None:
        logger.warning("mlflow uri has no @version; resolving 'latest' is not reproducible")

    try:
        local = _download_from_registry(name, ref)
        if local is None and ref is None:
            local = _download_from_run(name)
    except Exception as exc:
        raise ArtifactError(f"mlflow load failed for {location!r}: {exc}") from exc

    if local is None:
        raise ArtifactError(f"no mlflow artifacts found for model {name!r} (ref={ref!r})")

    layout = Path(local)
    if not (layout / "model.pkl").is_file() and (layout / _ARTIFACT_PATH).is_dir():
        layout = layout / _ARTIFACT_PATH
    return read_layout(layout)
