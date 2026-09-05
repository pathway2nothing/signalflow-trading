"""Env-driven log verbosity for every signalflow package, plus the ``step`` helper.

``SF_LOG_LEVEL`` (DEBUG/INFO/WARNING/...) wins; else ``SF_VERBOSE`` truthy
means DEBUG; else the default sink is quieted to INFO.

Convention across the core: one INFO line per finished operation the user
explicitly asked for (a fit, a backtest, a walk-forward), DEBUG for the steps
inside it (features, sampling, labels, folds, forecast slots, detectors, fills).
"""

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

_TRUTHY = {"1", "true", "yes", "on"}


def setup_logging() -> None:
    """Replace loguru's default DEBUG sink according to SF_LOG_LEVEL / SF_VERBOSE."""
    level = os.environ.get("SF_LOG_LEVEL", "").strip().upper()
    if not level:
        verbose = os.environ.get("SF_VERBOSE", "").strip().lower() in _TRUTHY
        level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level)


def frame_summary(frame) -> str:
    """``rows=N pairs=K span=a..b`` for a canonical (pair, ts) frame; safe on empty frames."""
    if frame is None or frame.height == 0:
        return "rows=0"
    parts = [f"rows={frame.height:,}"]
    if "pair" in frame.columns:
        parts.append(f"pairs={frame.get_column('pair').n_unique()}")
    if "ts" in frame.columns:
        ts = frame.get_column("ts")
        parts.append(f"span={ts.min()}..{ts.max()}")
    return " ".join(parts)


def names(items, limit: int = 6) -> str:
    """``[a, b, c, …+4]`` - a short bracketed preview of a name list."""
    items = list(items)
    shown = ", ".join(str(x) for x in items[:limit])
    more = f", …+{len(items) - limit}" if len(items) > limit else ""
    return f"[{shown}{more}]"


@contextmanager
def step(label: str, **fields) -> Iterator[dict]:
    """Emit one DEBUG line when the block ends: ``label: k=v k=v (1.23s)``.

    Fields passed up front and any added to the yielded dict inside the block are
    reported together, so a step can record what it produced. The line is
    attributed to the caller of ``with step(...)``. On an exception the line says
    ``failed`` and the exception propagates.
    """
    t0 = time.perf_counter()
    try:
        yield fields
    except Exception as exc:
        logger.opt(depth=2).debug(f"{label}: failed after {time.perf_counter() - t0:.2f}s ({type(exc).__name__}: {exc})")
        raise
    else:
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        body = f"{label}: {extras}" if extras else label
        logger.opt(depth=2).debug(f"{body} ({time.perf_counter() - t0:.2f}s)")
