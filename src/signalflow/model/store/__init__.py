"""Model-persistence layer."""


from signalflow.errors import ArtifactError
from signalflow.model.store.uri import resolve_uri

__all__ = ["save_model", "load_model", "resolve_uri"]


def _backend(scheme: str):
    if scheme == "file":
        from signalflow.model.store import local_store

        return local_store
    if scheme == "mlflow":
        from signalflow.model.store import mlflow_store

        return mlflow_store
    if scheme == "hf":
        from signalflow.model.store import hf_store

        return hf_store
    raise ArtifactError(f"no backend for scheme {scheme!r}")


def save_model(model, uri: str) -> str:
    """Persist a fitted ForecastModel to ``uri``; return the canonical uri."""
    scheme, location = resolve_uri(uri)
    return _backend(scheme).save(model, location)


def load_model(uri: str):
    """Load a ForecastModel from ``uri``."""
    scheme, location = resolve_uri(uri)
    return _backend(scheme).load(location)
