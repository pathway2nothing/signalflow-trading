"""
Rich progress output for SignalFlow backtests.

Provides visual feedback during long-running operations.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


console = Console()


@dataclass
class BacktestProgress:
    """Context manager for backtest progress display."""

    strategy_id: str
    verbose: bool = True
    _start_time: float = field(default=0.0, init=False)
    _stats: dict[str, Any] = field(default_factory=dict, init=False)

    def __enter__(self) -> "BacktestProgress":
        self._start_time = time.time()
        if self.verbose:
            console.print(
                Panel.fit(
                    f"[bold cyan]SignalFlow Backtest[/bold cyan]\nStrategy: [yellow]{self.strategy_id}[/yellow]",
                    border_style="cyan",
                )
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.time() - self._start_time
        if self.verbose and exc_type is None:
            self._print_summary(elapsed)

    def update_stage(self, stage: str) -> None:
        """Update current processing stage."""
        if self.verbose:
            console.print(f"  [dim]>[/dim] {stage}...")

    def set_stat(self, key: str, value: Any) -> None:
        """Set a statistic for final summary."""
        self._stats[key] = value

    def _print_summary(self, elapsed: float) -> None:
        """Print final summary table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        # Add all stats
        for key, value in self._stats.items():
            if isinstance(value, float):
                if "pct" in key.lower() or "rate" in key.lower():
                    formatted = f"{value:.1%}"
                elif abs(value) > 1000:
                    formatted = f"{value:,.2f}"
                else:
                    formatted = f"{value:.4f}"
            elif isinstance(value, int):
                formatted = f"{value:,}"
            else:
                formatted = str(value)
            table.add_row(key, formatted)

        table.add_row("Duration", f"{elapsed:.2f}s")

        console.print()
        console.print(
            Panel(
                table,
                title="[bold green]Backtest Complete[/bold green]",
                border_style="green",
            )
        )


@contextmanager
def backtest_progress(strategy_id: str, verbose: bool = True) -> Generator[BacktestProgress, None, None]:
    """Context manager for backtest progress tracking.

    Args:
        strategy_id: Name of the strategy being backtested
        verbose: Whether to print progress output

    Yields:
        BacktestProgress instance for updating progress

    Example:
        with backtest_progress("my_strategy") as progress:
            progress.update_stage("Loading data")
            # ... do work ...
            progress.update_stage("Detecting signals")
            # ... do work ...
            progress.set_stat("Trades", 42)
            progress.set_stat("Sharpe", 1.5)
    """
    progress = BacktestProgress(strategy_id=strategy_id, verbose=verbose)
    with progress:
        yield progress


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a Rich progress bar for iteration tracking.

    Returns:
        Progress instance ready for use with `track()` or `add_task()`

    Example:
        with create_progress_bar() as progress:
            task = progress.add_task("Processing bars", total=10000)
            for bar in bars:
                process(bar)
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_metrics(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metric names to values
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            if "sharpe" in key.lower() or "sortino" in key.lower():
                color = "green" if value > 1 else "yellow" if value > 0 else "red"
                formatted = f"[{color}]{value:.3f}[/{color}]"
            elif "drawdown" in key.lower():
                color = "green" if value > -0.1 else "yellow" if value > -0.2 else "red"
                formatted = f"[{color}]{value:.1%}[/{color}]"
            elif "rate" in key.lower() or "pct" in key.lower():
                formatted = f"{value:.1%}"
            else:
                formatted = f"{value:,.4f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

        table.add_row(key, formatted)

    console.print(table)


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


__all__ = [
    "console",
    "BacktestProgress",
    "backtest_progress",
    "create_progress_bar",
    "print_metrics",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
]
