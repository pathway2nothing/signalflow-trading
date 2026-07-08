"""Deterministic content hashing and source-code fingerprinting.

Shared home for the hashing primitives so neither :mod:`signalflow.model` nor
:mod:`signalflow.experiment` has to import the other for cache keys.
"""

import hashlib
import inspect
import json


def stable_hash(obj) -> str:
    """Deterministic content hash of a JSON-able object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:24]


def code_fingerprint(obj) -> str:
    """Hash the source of a callable/class so an edit to the producing code changes the key.

    Falls back to a module-version / qualified-name identity when source is not
    retrievable (e.g. C extensions, builtins).
    """
    target = obj

    if not (inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj)):
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
        return stable_hash(
            {"module": module, "qualname": getattr(target, "__qualname__", repr(target)), "version": version}
        )
