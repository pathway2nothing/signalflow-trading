"""CLI benchmark command for signalflow-ta Numba performance.

Runs a suite of technical indicators on synthetic data and reports timing.
"""

from __future__ import annotations

import importlib
import time
from datetime import datetime, timedelta
from typing import Any

import click
import numpy as np
import polars as pl

# ── Benchmark suite ──────────────────────────────────────────

BENCHMARK_SUITE: list[tuple[str, str, dict[str, Any], str]] = [
    # (module.ClassName, display_name, params, category)
    ("signalflow.ta.momentum.core.RsiMom", "RSI", {"period": 14}, "Momentum"),
    ("signalflow.ta.momentum.core.CmoMom", "CMO", {"period": 14}, "Momentum"),
    (
        "signalflow.ta.momentum.oscillators.StochMom",
        "Stochastic",
        {"k_period": 14, "d_period": 3, "smooth_k": 3},
        "Momentum",
    ),
    ("signalflow.ta.momentum.oscillators.CciMom", "CCI", {"period": 20}, "Momentum"),
    ("signalflow.ta.overlap.adaptive.JmaSmooth", "JMA", {"period": 7, "phase": 0}, "Overlap"),
    ("signalflow.ta.overlap.adaptive.KamaSmooth", "KAMA", {"period": 10, "fast": 2, "slow": 30}, "Overlap"),
    ("signalflow.ta.overlap.adaptive.FramaSmooth", "FRAMA", {"period": 16}, "Overlap"),
    ("signalflow.ta.overlap.smoothers.WmaSmooth", "WMA", {"period": 20}, "Overlap"),
    ("signalflow.ta.overlap.smoothers.HmaSmooth", "HMA", {"period": 20}, "Overlap"),
    ("signalflow.ta.trend.strength.AdxTrend", "ADX", {"period": 14}, "Trend"),
    ("signalflow.ta.trend.stops.PsarTrend", "PSAR", {"af_step": 0.02, "af_max": 0.2}, "Trend"),
    ("signalflow.ta.trend.stops.SupertrendTrend", "Supertrend", {"period": 10, "multiplier": 3.0}, "Trend"),
    ("signalflow.ta.volatility.bands.BollingerVol", "Bollinger", {"period": 20, "std_dev": 2.0}, "Volatility"),
    ("signalflow.ta.volatility.range.AtrVol", "ATR", {"period": 14}, "Volatility"),
]


# ── Helpers ──────────────────────────────────────────────────


def _generate_ohlcv(n_rows: int) -> pl.DataFrame:
    """Generate synthetic OHLCV data for benchmarking."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]

    base_price = 100.0
    returns = rng.normal(0, 0.02, n_rows)
    close = base_price * np.exp(np.cumsum(returns))

    open_prices = np.empty(n_rows)
    open_prices[0] = close[0]
    open_prices[1:] = close[:-1]

    high = np.maximum(open_prices, close) * (1 + np.abs(rng.normal(0, 0.01, n_rows)))
    low = np.minimum(open_prices, close) * (1 - np.abs(rng.normal(0, 0.01, n_rows)))
    volume = np.abs(rng.normal(1000, 300, n_rows))

    return pl.DataFrame({
        "pair": ["BTCUSDT"] * n_rows,
        "timestamp": timestamps,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _load_class(class_path: str) -> type:
    """Dynamically import a class from dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def _time_indicator(cls: type, params: dict[str, Any], df: pl.DataFrame, runs: int) -> float:
    """Time indicator computation (median of N runs). First call is warmup."""
    indicator = cls(**params)
    # Warmup (JIT compilation on first call)
    indicator.compute_pair(df)

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        indicator.compute_pair(df)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


# ── CLI Command ──────────────────────────────────────────────


@click.command("benchmark-ta")
@click.option("--rows", "-n", default=100_000, show_default=True, help="Number of data rows.")
@click.option("--runs", "-r", default=5, show_default=True, help="Timing iterations per indicator.")
@click.option("--verbose", "-v", is_flag=True, help="Show additional details.")
def benchmark_ta(rows: int, runs: int, verbose: bool) -> None:
    """Benchmark signalflow-ta technical indicators.

    \b
    Runs a suite of Numba-accelerated indicators on synthetic OHLCV data
    and reports median execution time.

    \b
    Examples:
        sf benchmark-ta
        sf benchmark-ta --rows 500000
        sf benchmark-ta --rows 10000 --runs 3 --verbose
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        click.secho("Error: rich is required for benchmark output.", fg="red", err=True)
        raise SystemExit(1)  # noqa: B904

    try:
        from signalflow.ta._numba_compat import NUMBA_AVAILABLE
    except ImportError:
        click.secho("Error: signalflow-ta is not installed.", fg="red", err=True)
        raise SystemExit(1)  # noqa: B904

    console = Console()

    # Header
    console.print()
    console.print("[bold]SignalFlow TA Benchmark[/bold]")
    numba_status = "[green]ON[/green]" if NUMBA_AVAILABLE else "[yellow]OFF (Python fallback)[/yellow]"
    console.print(f"  Rows: {rows:,}  |  Runs: {runs}  |  Numba: {numba_status}")
    console.print()

    # Generate data
    with console.status("[bold]Generating synthetic OHLCV data..."):
        df = _generate_ohlcv(rows)

    # Build results table
    table = Table(title=f"Indicator Benchmark ({rows:,} rows, median of {runs} runs)")
    table.add_column("Indicator", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Rows/sec", justify="right")
    table.add_column("Status", justify="center")

    total_time = 0.0
    passed = 0
    failed = 0

    for class_path, name, params, category in BENCHMARK_SUITE:
        try:
            cls = _load_class(class_path)
            t = _time_indicator(cls, params, df, runs)
            total_time += t

            ms = t * 1000
            rows_per_sec = int(rows / t) if t > 0 else 0

            if ms < 100:
                status = "[green]FAST[/green]"
            elif ms < 500:
                status = "[yellow]OK[/yellow]"
            else:
                status = "[red]SLOW[/red]"

            table.add_row(name, category, f"{ms:.1f}", f"{rows_per_sec:,}", status)
            passed += 1

            if verbose:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                console.print(f"  [dim]{name}({param_str}) = {ms:.1f}ms[/dim]")

        except Exception as e:
            table.add_row(name, category, "—", "—", f"[red]ERR: {e}[/red]")
            failed += 1

    console.print(table)
    console.print()

    # Summary
    total_ms = total_time * 1000
    console.print(f"[bold]Total:[/bold] {total_ms:.0f}ms for {passed} indicators  |  ", end="")
    if failed:
        console.print(f"[red]{failed} failed[/red]")
    else:
        console.print("[green]All passed[/green]")
    console.print()
