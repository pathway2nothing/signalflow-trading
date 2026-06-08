"""Warmup contract: warmup window as a declared, derivable quantity (§4.2).

Warmup-invariance is enforced per feature (``Feature.warmup`` /
``assert_reproducible`` in ``feature/base.py``) and ``forecast_window`` pins the
forecast lookback. This module lifts the same idea to the *system* level: any
component can declare how many bars of warmup it needs, and a runner derives the
required window as the max over its components — instead of a magic constant.

A component declares warmup via either a ``warmup_bars`` attribute/method or the
feature-style ``warmup`` property. Using the SAME derived window in backtest and
live is what keeps recursive indicators from diverging at the live cold-start.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def warmup_bars_of(component: Any) -> int:
    """Return the warmup-bar requirement a single component declares.

    Precedence: ``warmup_bars`` (int or zero-arg callable) → ``warmup``
    (int or property) → 0. Unknown / None components contribute 0.
    """
    if component is None:
        return 0
    for attr in ("warmup_bars", "warmup"):
        if hasattr(component, attr):
            value = getattr(component, attr)
            if callable(value):
                value = value()
            if value is not None:
                return int(value)
    return 0


def required_warmup_bars(*components: Any, floor: int = 0) -> int:
    """Derive the warmup window for a set of components (max of declarations).

    Iterables of components are flattened one level, so you can pass lists of
    detectors/features/rules directly. ``floor`` is a lower bound (e.g. a
    runner's configured minimum).

    Returns:
        ``max(floor, max(warmup_bars_of(c) for all c))``.
    """
    best = floor
    for component in components:
        if isinstance(component, Iterable) and not isinstance(component, (str, bytes)):
            for sub in component:
                best = max(best, warmup_bars_of(sub))
        else:
            best = max(best, warmup_bars_of(component))
    return best


def assert_warmup_consistency(backtest_bars: int, live_bars: int) -> None:
    """Assert backtest and live use the same warmup window (parity contract).

    A different warmup slice between modes silently changes recursive-indicator
    values, breaking research-to-production parity — so a mismatch raises rather
    than degrading quietly.

    Raises:
        ValueError: if the two warmup windows differ.
    """
    if backtest_bars != live_bars:
        raise ValueError(
            f"Warmup window mismatch breaks parity: backtest={backtest_bars} bars "
            f"vs live={live_bars} bars. Both modes must warm up over the same slice."
        )
