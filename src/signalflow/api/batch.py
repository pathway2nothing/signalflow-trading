"""Batch run pipeline -- execute multiple FlowBuilder configs and compare.

Usage::

    from signalflow.api.batch import batch_run

    configs = [
        sf.flow().data(...).detector("sma_cross", fast=10, slow=30),
        sf.flow().data(...).detector("sma_cross", fast=20, slow=50),
        sf.flow().data(...).detector("rsi_reversal"),
    ]
    results = batch_run(configs)
    print(results.comparison.summary())
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from signalflow.analytic.compare import ComparisonResult
    from signalflow.api.flow import FlowBuilder, FlowResult


@dataclass
class BatchResult:
    """Container for batch run results.

    Attributes:
        results: List of FlowResult, one per config (order preserved).
        labels: Human-readable label per config.
        errors: Dict mapping index to exception for failed runs.
        elapsed: Total wall-clock time in seconds.
    """

    results: list[FlowResult | None]
    labels: list[str]
    errors: dict[int, Exception] = field(default_factory=dict)
    elapsed: float = 0.0

    @property
    def successful(self) -> list[FlowResult]:
        """Return only successful (non-None) results."""
        return [r for r in self.results if r is not None]

    @property
    def comparison(self) -> ComparisonResult:
        """Compare all successful results.

        Returns:
            ComparisonResult with side-by-side metrics.

        Raises:
            ValueError: If fewer than 2 results succeeded.
        """
        from signalflow.analytic.compare import compare_results

        good = self.successful
        if len(good) < 2:
            msg = f"Need at least 2 successful results to compare, got {len(good)}"
            raise ValueError(msg)
        return compare_results(*good)

    def summary(self) -> str:
        """Return a text summary of the batch run."""
        lines: list[str] = []
        lines.append(f"Batch Run: {len(self.results)} configs, "
                     f"{len(self.successful)} succeeded, "
                     f"{len(self.errors)} failed, "
                     f"{self.elapsed:.1f}s")
        if len(self.successful) >= 2:
            lines.append("")
            lines.append(self.comparison.summary())
        return "\n".join(lines)


ProgressCallback = Callable[[int, int, str], None]
"""Signature: (completed_count, total_count, label) -> None"""


def batch_run(
    configs: list[FlowBuilder],
    *,
    labels: list[str] | None = None,
    parallel: bool = False,
    max_workers: int = 4,
    progress: ProgressCallback | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> BatchResult:
    """Run multiple FlowBuilder configs and collect results.

    Args:
        configs: List of pre-configured FlowBuilder instances.
        labels: Optional human-readable labels (one per config).
            If not provided, auto-generates "run_0", "run_1", etc.
        parallel: If True, run configs in parallel threads.
        max_workers: Max thread count when parallel=True.
        progress: Optional callback ``(completed, total, label) -> None``.
        run_kwargs: Extra keyword arguments passed to each ``FlowBuilder.run()``.

    Returns:
        BatchResult with results, comparison, and error tracking.
    """
    if not configs:
        msg = "batch_run requires at least 1 config"
        raise ValueError(msg)

    n = len(configs)
    if labels is None:
        labels = [_auto_label(cfg, i) for i, cfg in enumerate(configs)]
    elif len(labels) != n:
        msg = f"labels length ({len(labels)}) != configs length ({n})"
        raise ValueError(msg)

    kwargs = run_kwargs or {}
    results: list[FlowResult | None] = [None] * n
    errors: dict[int, Exception] = {}

    t0 = time.monotonic()

    if parallel and n > 1:
        _run_parallel(configs, labels, results, errors, kwargs, max_workers, progress)
    else:
        _run_sequential(configs, labels, results, errors, kwargs, progress)

    elapsed = time.monotonic() - t0

    # Assign strategy_id to results for comparison naming
    for res, lbl in zip(results, labels, strict=True):
        if res is not None and hasattr(res, "flow_config") and res.flow_config is not None:
            res.flow_config.strategy_id = lbl  # type: ignore[union-attr]

    return BatchResult(results=results, labels=labels, errors=errors, elapsed=elapsed)


# ── Internal runners ───────────────────────────────────────────────────


def _auto_label(cfg: FlowBuilder, index: int) -> str:
    """Generate a label from the FlowBuilder config."""
    if hasattr(cfg, "_strategy_id") and cfg._strategy_id:
        return str(cfg._strategy_id)
    return f"run_{index}"


def _run_one(cfg: FlowBuilder, kwargs: dict[str, Any]) -> FlowResult:
    """Run a single FlowBuilder config."""
    return cfg.run(**kwargs)


def _run_sequential(
    configs: list[FlowBuilder],
    labels: list[str],
    results: list[FlowResult | None],
    errors: dict[int, Exception],
    kwargs: dict[str, Any],
    progress: ProgressCallback | None,
) -> None:
    for i, cfg in enumerate(configs):
        try:
            results[i] = _run_one(cfg, kwargs)
        except Exception as e:
            logger.warning(f"batch_run: config {i} ({labels[i]}) failed: {e}")
            errors[i] = e
        if progress is not None:
            progress(i + 1, len(configs), labels[i])


def _run_parallel(
    configs: list[FlowBuilder],
    labels: list[str],
    results: list[FlowResult | None],
    errors: dict[int, Exception],
    kwargs: dict[str, Any],
    max_workers: int,
    progress: ProgressCallback | None,
) -> None:
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(configs))) as pool:
        future_to_idx = {pool.submit(_run_one, cfg, kwargs): i for i, cfg in enumerate(configs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"batch_run: config {idx} ({labels[idx]}) failed: {e}")
                errors[idx] = e
            completed += 1
            if progress is not None:
                progress(completed, len(configs), labels[idx])
