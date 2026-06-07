"""Pinned-inference model delivery: declarative refs + lazy resolution.

Forecast models are trained elsewhere and arrive in the trading pipeline as
versioned, reproducible artifacts. This package provides:

    - :class:`ModelRef` — declarative, versioned pointer to an artifact.
    - :class:`Resolver` / :class:`MlflowResolver` — lazy loading of weights.
    - :class:`ModelRegistry` / :class:`CachingModelRegistry` — fetch-by-ref port.

Importing this package does NOT require mlflow; weights load only on resolve.
"""

from __future__ import annotations

from .model_ref import ModelRef
from .registry import CachingModelRegistry, ModelRegistry
from .resolver import MlflowResolver, Resolver

__all__ = [
    "CachingModelRegistry",
    "MlflowResolver",
    "ModelRef",
    "ModelRegistry",
    "Resolver",
]
