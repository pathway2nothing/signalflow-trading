"""ModelRegistry — port + caching implementation over a :class:`Resolver`.

The :class:`ModelRegistry` protocol is the consumer-facing port for fetching
models by :class:`ModelRef`. :class:`CachingModelRegistry` is a simple,
lazy, in-process implementation: a ref is resolved at most once and reused
from cache thereafter.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from loguru import logger

from .model_ref import ModelRef
from .resolver import Resolver


@runtime_checkable
class ModelRegistry(Protocol):
    """Port: fetch loaded models by ModelRef."""

    def get(self, ref: ModelRef) -> Any:
        """Return the loaded model for ``ref`` (resolving lazily if needed)."""
        ...

    def has(self, ref: ModelRef) -> bool:
        """Return True if ``ref`` is already loaded/cached."""
        ...


class CachingModelRegistry:
    """In-process registry that lazily resolves and caches models by ModelRef.

    Holds a :class:`Resolver` and a cache keyed by ModelRef (which is frozen and
    hashable). The first :meth:`get` for a ref triggers resolution; subsequent
    calls return the cached artifact without re-resolving.

    Args:
        resolver: The Resolver used to load uncached refs.
    """

    def __init__(self, resolver: Resolver) -> None:
        self._resolver = resolver
        self._cache: dict[ModelRef, Any] = {}

    def get(self, ref: ModelRef) -> Any:
        """Return the model for ``ref``, resolving and caching on first access."""
        if ref in self._cache:
            return self._cache[ref]
        logger.debug(f"CachingModelRegistry: cache miss for {ref.uri}, resolving")
        model = self._resolver.resolve(ref)
        self._cache[ref] = model
        return model

    def has(self, ref: ModelRef) -> bool:
        """Return True if ``ref`` is already cached (no resolution triggered)."""
        return ref in self._cache
