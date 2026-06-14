"""
Content-addressed artifact cache.

Caches polars frames on disk as parquet, keyed by a content hash that folds in a
**code fingerprint** of the producing callable/class. Editing a transform's body
changes its source, which changes the fingerprint, which changes the key - so a
stale column can never be silently reused.

Keys also carry a ``kind`` discriminator so out-of-sample-for-training artifacts
(``kind="oos_for_training"``) and production-inference artifacts
(``kind="inference"``) can never collide.
"""


import inspect
from pathlib import Path
from typing import Callable

import polars as pl

from signalflow.errors import ArtifactError
from signalflow.model.oos import stable_hash

VALID_KINDS = ("oos_for_training", "inference")


def code_fingerprint(obj) -> str:
    """
    Hash the source of a callable/class (or its module version) so an edit to
    the producing code changes the cache key.

    Falls back to a module-version / qualified-name identity when source is not
    retrievable (e.g. C extensions, builtins).
    """
    target = obj

    if not (inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj)):
        if callable(obj) and not inspect.isbuiltin(obj):
            target = type(obj)
        else:
            target = type(obj)
    try:
        src = inspect.getsource(target)
        return stable_hash({"src": src})
    except (OSError, TypeError):
        module = getattr(target, "__module__", "?")
        version = "?"
        try:
            import importlib

            version = getattr(importlib.import_module(module), "__version__", "?")
        except Exception:
            pass
        return stable_hash({"module": module, "qualname": getattr(target, "__qualname__", repr(target)), "version": version})


class ArtifactCache:
    """Disk cache of polars frames addressed by a content + code-fingerprint key."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)


    def key(self, parts: dict, *, producer=None, kind: str = "inference") -> str:
        """Compute the content-addressed key for ``parts``."""
        if kind not in VALID_KINDS:
            raise ArtifactError(f"unknown artifact kind {kind!r}; expected one of {VALID_KINDS}")
        material = {
            "kind": kind,
            "parts": parts,
            "code": code_fingerprint(producer) if producer is not None else None,
        }
        return stable_hash(material)

    def _path(self, key: str) -> Path:

        safe = key.replace(":", "_")
        return self.root / f"{safe}.parquet"


    def get(self, key: str) -> pl.DataFrame | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as exc:
            raise ArtifactError(f"failed to read cached artifact {path}: {exc}") from exc

    def put(self, key: str, df: pl.DataFrame) -> None:
        if not isinstance(df, pl.DataFrame):
            raise ArtifactError(f"ArtifactCache stores polars DataFrames, got {type(df)!r}")
        tmp = self._path(key).with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(self._path(key))


    def compute_cached(self, key: str, fn: Callable[[], pl.DataFrame]) -> pl.DataFrame:
        """Return the cached frame for ``key`` or compute it via ``fn`` and store it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        df = fn()
        self.put(key, df)
        return df
