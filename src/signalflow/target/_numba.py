"""
Optional-numba shim.

The legacy labelers were Numba-accelerated. Numba is not a hard dependency of
the V5 framework, so this module provides a drop-in ``njit`` / ``prange`` that
works whether or not numba is installed. When numba is present the real
decorators are used (JIT + parallelism); when it is absent the decorated
function runs as plain Python so ``import signalflow`` never breaks.

Usage (identical to numba)::

from signalflow.target._numba import njit, prange

@njit(parallel=True, cache=True)
    def f(...):
        for i in prange(n):
            ...
"""


try:
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in for :func:`numba.njit`."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator


__all__ = ["njit", "prange", "HAS_NUMBA"]
