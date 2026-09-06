"""Model-persistence layer."""


from signalflow.errors import ArtifactError
from signalflow.model.store.uri import resolve_uri

__all__ = ["load_model", "resolve_uri", "save_model"]


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


def load_model(uri: str, trust_remote: bool = False):
    """Load a ForecastModel from ``uri``; ``hf://`` artifacts need ``trust_remote=True`` (remote code)."""
    scheme, location = resolve_uri(uri)
    if scheme == "hf":
        return _backend(scheme).load(location, trust_remote=trust_remote)
    return _backend(scheme).load(location)
