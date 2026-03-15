"""Result comparison utility — compare N backtest/flow results side by side.

Usage::

    from signalflow.analytic import compare_results

    comparison = compare_results(result_a, result_b, result_c)
    print(comparison.summary())
    print(comparison.best("sharpe_ratio"))
    df = comparison.metrics_table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class _HasMetrics(Protocol):
    """Protocol for objects that expose a ``metrics`` dict."""

    @property
    def metrics(self) -> dict[str, float]: ...


@runtime_checkable
class _HasName(Protocol):
    """Protocol for objects that expose a name/strategy_id."""

    @property
    def strategy_id(self) -> str: ...


def _result_name(result: Any, index: int) -> str:
    """Extract a human-readable name from a result object."""
    if hasattr(result, "strategy_id") and result.strategy_id:
        return str(result.strategy_id)
    if hasattr(result, "name") and result.name:
        return str(result.name)
    if hasattr(result, "config") and isinstance(result.config, dict):
        sid = result.config.get("strategy_id", "")
        if sid:
            return str(sid)
    return f"run_{index}"


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple backtest results.

    Attributes
    ----------
    names:
        Human-readable identifiers for each result.
    metrics_table:
        Polars DataFrame with rows = metrics, columns = results.
    raw_metrics:
        List of raw metric dicts (one per result).
    """

    names: list[str]
    metrics_table: pl.DataFrame
    raw_metrics: list[dict[str, float]] = field(repr=False)

    # ── Query helpers ──────────────────────────────────────────────

    def best(self, metric: str) -> str:
        """Return the name of the result with the highest value for *metric*.

        Parameters
        ----------
        metric:
            Metric key, e.g. ``"sharpe_ratio"``, ``"total_return"``.

        Returns
        -------
        str
            Name of the best-performing result for that metric.

        Raises
        ------
        KeyError
            If *metric* not found in the comparison table.
        """
        if metric not in self.metrics_table["metric"].to_list():
            available = ", ".join(self.metrics_table["metric"].to_list()[:10])
            msg = f"Metric {metric!r} not found. Available: {available} ..."
            raise KeyError(msg)

        row = self.metrics_table.filter(pl.col("metric") == metric)
        values: dict[str, float] = {}
        for name in self.names:
            val = row[name].item()
            if val is not None:
                values[name] = float(val)
        if not values:
            return self.names[0]
        return max(values, key=lambda k: values[k])

    def worst(self, metric: str) -> str:
        """Return the name of the result with the lowest value for *metric*."""
        if metric not in self.metrics_table["metric"].to_list():
            available = ", ".join(self.metrics_table["metric"].to_list()[:10])
            msg = f"Metric {metric!r} not found. Available: {available} ..."
            raise KeyError(msg)

        row = self.metrics_table.filter(pl.col("metric") == metric)
        values: dict[str, float] = {}
        for name in self.names:
            val = row[name].item()
            if val is not None:
                values[name] = float(val)
        if not values:
            return self.names[0]
        return min(values, key=lambda k: values[k])

    # ── Export ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted text summary of the comparison."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  Result Comparison")
        lines.append("=" * 60)

        # Header
        header = f"{'Metric':<25}"
        for name in self.names:
            header += f"  {name:>14}"
        lines.append(header)
        lines.append("-" * len(header))

        # Key metrics first
        priority = [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "n_trades",
            "sortino_ratio",
            "calmar_ratio",
        ]
        all_metrics = self.metrics_table["metric"].to_list()
        ordered = [m for m in priority if m in all_metrics]
        ordered += [m for m in all_metrics if m not in ordered]

        for metric_name in ordered:
            row = self.metrics_table.filter(pl.col("metric") == metric_name)
            line = f"{metric_name:<25}"
            for name in self.names:
                val = row[name].item()
                if val is None:
                    line += f"  {'—':>14}"
                else:
                    line += f"  {float(val):>14.4f}"
            lines.append(line)

        lines.append("-" * len(header))

        # Best per key metric
        lines.append("")
        lines.append("Best per metric:")
        for m in priority:
            if m in all_metrics:
                lines.append(f"  {m:<25} → {self.best(m)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export comparison as a JSON-serializable dict."""
        return {
            "names": self.names,
            "metrics": {
                name: metrics for name, metrics in zip(self.names, self.raw_metrics, strict=True)
            },
        }


def compare_results(*results: Any) -> ComparisonResult:
    """Compare multiple backtest or flow results.

    Accepts :class:`BacktestResult`, :class:`FlowResult`, or any object
    with a ``.metrics`` property returning ``dict[str, float]``.

    Parameters
    ----------
    *results:
        Two or more result objects to compare.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ValueError
        If fewer than 2 results are provided.
    TypeError
        If a result object doesn't have a ``metrics`` property.

    Example
    -------
    ::

        from signalflow.analytic import compare_results

        cmp = compare_results(result_a, result_b)
        print(cmp.summary())
        print(f"Best Sharpe: {cmp.best('sharpe_ratio')}")
    """
    if len(results) < 2:
        msg = f"compare_results requires at least 2 results, got {len(results)}"
        raise ValueError(msg)

    names: list[str] = []
    all_metrics: list[dict[str, float]] = []

    for i, result in enumerate(results):
        name = _result_name(result, i)
        # Deduplicate names
        if name in names:
            name = f"{name}_{i}"
        names.append(name)

        # Extract metrics
        if hasattr(result, "metrics") and callable(getattr(type(result), "metrics", None)):
            # It's a property
            metrics = result.metrics
        elif hasattr(result, "metrics"):
            metrics = result.metrics
        elif hasattr(result, "backtest_metrics") and result.backtest_metrics is not None:
            # FlowResult with backtest metrics
            metrics = dict(result.backtest_metrics)
        else:
            msg = f"Result {i} ({type(result).__name__}) has no .metrics attribute"
            raise TypeError(msg)

        if not isinstance(metrics, dict):
            msg = f"Result {i} .metrics returned {type(metrics).__name__}, expected dict"
            raise TypeError(msg)

        all_metrics.append(metrics)

    # Collect all metric keys across results
    all_keys: list[str] = []
    seen: set[str] = set()
    for m in all_metrics:
        for k in m:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Build Polars DataFrame
    data: dict[str, list[float | None]] = {"metric": all_keys}  # type: ignore[dict-item]
    for name, metrics in zip(names, all_metrics, strict=True):
        data[name] = [metrics.get(k) for k in all_keys]

    table = pl.DataFrame(data)

    return ComparisonResult(
        names=names,
        metrics_table=table,
        raw_metrics=all_metrics,
    )
