"""Resolver — lazily turns a :class:`ModelRef` into a loaded model artifact.

The :class:`Resolver` protocol is the loading port; :class:`MlflowResolver` is
the MLflow-backed implementation. Crucially, ``mlflow`` is imported *inside*
``resolve`` so that ``import signalflow.models`` works without mlflow installed
— weights are only fetched on an explicit ``resolve`` call.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from loguru import logger

from .model_ref import ModelRef


@runtime_checkable
class Resolver(Protocol):
    """Port: resolve a ModelRef into a loaded, ready-to-use model object."""

    def resolve(self, ref: ModelRef) -> Any:
        """Load and return the artifact referenced by ``ref``."""
        ...


class MlflowResolver:
    """Resolver backed by the MLflow Model Registry.

    Loading is fully lazy: ``mlflow`` is imported only when :meth:`resolve` is
    called, and the underlying loader is isolated in :meth:`_load` so tests can
    override it without a real MLflow server.

    Args:
        tracking_uri: Optional MLflow tracking URI. If set, applied on the first
            resolve call.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri

    def resolve(self, ref: ModelRef) -> Any:
        """Resolve ``ref`` to a loaded MLflow model.

        Builds the URI ``models:/{ref.name}/{ref.version}`` and delegates the
        actual load to :meth:`_load`.

        Raises:
            ValueError: If ``ref.source`` is not ``"mlflow"``.
        """
        if ref.source != "mlflow":
            raise ValueError(f"MlflowResolver cannot resolve source={ref.source!r}; expected 'mlflow'")
        uri = ref.uri
        logger.debug(f"MlflowResolver: resolving {uri}")
        return self._load(uri)

    def _load(self, uri: str) -> Any:
        """Load a model from an MLflow ``models:/`` URI (lazy mlflow import).

        Overridable for testing. Uses the generic pyfunc loader so any model
        flavor registered under the URI can be loaded.
        """
        import mlflow.pyfunc  # type: ignore[import-not-found]

        if self.tracking_uri is not None:
            import mlflow  # type: ignore[import-not-found]

            mlflow.set_tracking_uri(self.tracking_uri)
        return mlflow.pyfunc.load_model(uri)
