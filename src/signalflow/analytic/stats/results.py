"""Result dataclasses for statistical validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import plotly.graph_objects as go


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence interval for a metric.

    Attributes:
        metric_name: Name of the metric
        point_estimate: Original computed value
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
        confidence_level: Confidence level (e.g., 0.95)
        method: Bootstrap method used ("bca", "percentile", "block")
    """

    metric_name: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str

    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper

    def is_significant(self, benchmark: float = 0.0) -> bool:
        """Check if interval excludes the benchmark value."""
        return not self.contains(benchmark)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo trade simulation.

    Monte Carlo simulation shuffles trade execution order to estimate
    the distribution of possible outcomes under different trade sequences.

    Attributes:
        n_simulations: Number of simulations performed
        final_equity_dist: Distribution of final equity values
        max_drawdown_dist: Distribution of maximum drawdowns
        max_consecutive_losses_dist: Distribution of max consecutive losing trades
        longest_drawdown_duration_dist: Distribution of longest drawdown durations
        equity_percentiles: Percentile values for final equity
        drawdown_percentiles: Percentile values for max drawdown
        risk_of_ruin: Probability of hitting the ruin threshold
        ruin_threshold: Drawdown threshold for risk of ruin calculation
        expected_max_drawdown: Mean of max drawdown distribution
        expected_worst_equity: 5th percentile of equity distribution
        original_final_equity: Final equity from actual backtest
        original_max_drawdown: Max drawdown from actual backtest
    """

    n_simulations: int

    # Distributions (numpy arrays)
    final_equity_dist: np.ndarray
    max_drawdown_dist: np.ndarray
    max_consecutive_losses_dist: np.ndarray
    longest_drawdown_duration_dist: np.ndarray

    # Percentiles
    equity_percentiles: dict[float, float]
    drawdown_percentiles: dict[float, float]

    # Risk metrics
    risk_of_ruin: float
    ruin_threshold: float
    expected_max_drawdown: float
    expected_worst_equity: float

    # Original values for comparison
    original_final_equity: float
    original_max_drawdown: float

    def to_dataframe(self) -> pl.DataFrame:
        """Export simulation results as Polars DataFrame."""
        return pl.DataFrame(
            {
                "final_equity": self.final_equity_dist,
                "max_drawdown": self.max_drawdown_dist,
                "max_consecutive_losses": self.max_consecutive_losses_dist,
                "longest_drawdown_duration": self.longest_drawdown_duration_dist,
            }
        )

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "Monte Carlo Simulation Results",
            "=" * 40,
            f"Simulations: {self.n_simulations:,}",
            "",
            "Final Equity Distribution:",
        ]

        for pct, val in sorted(self.equity_percentiles.items()):
            lines.append(f"  {pct * 100:5.1f}th percentile: ${val:,.2f}")

        lines.extend(
            [
                f"  Original:           ${self.original_final_equity:,.2f}",
                "",
                "Max Drawdown Distribution:",
            ]
        )

        for pct, val in sorted(self.drawdown_percentiles.items()):
            lines.append(f"  {pct * 100:5.1f}th percentile: {val * 100:.2f}%")

        lines.extend(
            [
                f"  Original:           {self.original_max_drawdown * 100:.2f}%",
                "",
                "Risk Metrics:",
                f"  Risk of Ruin (DD > {self.ruin_threshold * 100:.0f}%): {self.risk_of_ruin * 100:.2f}%",
                f"  Expected Max Drawdown: {self.expected_max_drawdown * 100:.2f}%",
                f"  Expected Worst Equity (P5): ${self.expected_worst_equity:,.2f}",
            ]
        )

        return "\n".join(lines)

    def plot(self) -> list[go.Figure]:
        """Generate Monte Carlo simulation plots."""
        from signalflow.analytic.stats.visualization import plot_monte_carlo

        return plot_monte_carlo(self)


@dataclass
class BootstrapResult:
    """Results from bootstrap confidence interval estimation.

    Attributes:
        n_bootstrap: Number of bootstrap resamples
        method: Bootstrap method ("bca", "percentile", "block")
        intervals: Confidence intervals for each metric
        distributions: Raw bootstrap distributions for each metric
        block_size: Block size for block bootstrap (None for IID)
    """

    n_bootstrap: int
    method: str
    intervals: dict[str, ConfidenceInterval]
    distributions: dict[str, np.ndarray]
    block_size: int | None = None

    def to_dataframe(self) -> pl.DataFrame:
        """Export confidence intervals as Polars DataFrame."""
        rows = []
        for name, ci in self.intervals.items():
            rows.append(
                {
                    "metric": name,
                    "point_estimate": ci.point_estimate,
                    "lower": ci.lower,
                    "upper": ci.upper,
                    "confidence_level": ci.confidence_level,
                    "method": ci.method,
                }
            )
        return pl.DataFrame(rows)

    def is_significant(self, metric: str, benchmark: float = 0.0) -> bool:
        """Check if metric is significantly different from benchmark."""
        if metric not in self.intervals:
            raise KeyError(f"Metric '{metric}' not found in results")
        return self.intervals[metric].is_significant(benchmark)

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "Bootstrap Confidence Intervals",
            "=" * 40,
            f"Method: {self.method.upper()}",
            f"Resamples: {self.n_bootstrap:,}",
        ]

        if self.block_size is not None:
            lines.append(f"Block size: {self.block_size}")

        lines.append("")

        for name, ci in sorted(self.intervals.items()):
            sig = "*" if ci.is_significant(0.0) else ""
            lines.append(
                f"{name}: {ci.point_estimate:.4f} "
                f"[{ci.lower:.4f}, {ci.upper:.4f}] "
                f"({ci.confidence_level * 100:.0f}% CI){sig}"
            )

        lines.append("")
        lines.append("* = significantly different from 0")

        return "\n".join(lines)

    def plot(self) -> go.Figure:
        """Generate bootstrap confidence interval plot."""
        from signalflow.analytic.stats.visualization import plot_bootstrap

        return plot_bootstrap(self)


@dataclass
class StatisticalTestResult:
    """Results from statistical significance tests.

    Implements Probabilistic Sharpe Ratio (PSR) and Minimum Track Record
    Length (MinTRL) based on Bailey & Lopez de Prado (2012).

    Attributes:
        psr: Probabilistic Sharpe Ratio (probability SR > benchmark)
        psr_benchmark: Benchmark Sharpe ratio used
        psr_is_significant: Whether PSR indicates significance
        min_track_record_length: Minimum trades needed for significance
        current_track_record: Current number of observations
        track_record_sufficient: Whether current track record is sufficient
        deflated_sr: Deflated Sharpe Ratio (optional, for multiple testing)
    """

    psr: float | None = None
    psr_benchmark: float = 0.0
    psr_is_significant: bool = False

    min_track_record_length: int | None = None
    current_track_record: int = 0
    track_record_sufficient: bool = False

    deflated_sr: float | None = None

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "Statistical Significance Tests",
            "=" * 40,
        ]

        if self.psr is not None:
            sig_str = "YES" if self.psr_is_significant else "NO"
            lines.extend(
                [
                    "Probabilistic Sharpe Ratio (PSR):",
                    f"  P(SR > {self.psr_benchmark:.2f}): {self.psr * 100:.2f}%",
                    f"  Statistically significant: {sig_str}",
                    "",
                ]
            )

        if self.min_track_record_length is not None:
            suff_str = "YES" if self.track_record_sufficient else "NO"
            lines.extend(
                [
                    "Minimum Track Record Length:",
                    f"  Current observations: {self.current_track_record}",
                    f"  Minimum required: {self.min_track_record_length}",
                    f"  Sufficient: {suff_str}",
                ]
            )
        else:
            lines.extend(
                [
                    "Minimum Track Record Length:",
                    f"  Current observations: {self.current_track_record}",
                    "  Cannot compute MinTRL (SR <= benchmark)",
                ]
            )

        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Combined validation results from all statistical analyses.

    Attributes:
        monte_carlo: Monte Carlo simulation results
        bootstrap: Bootstrap confidence interval results
        statistical_tests: Statistical significance test results
    """

    monte_carlo: MonteCarloResult | None = None
    bootstrap: BootstrapResult | None = None
    statistical_tests: StatisticalTestResult | None = None

    def summary(self) -> str:
        """Return comprehensive summary of all validation results."""
        lines = [
            "Comprehensive Statistical Validation",
            "=" * 50,
            "",
        ]

        if self.monte_carlo:
            lines.append(self.monte_carlo.summary())
            lines.append("")

        if self.bootstrap:
            lines.append(self.bootstrap.summary())
            lines.append("")

        if self.statistical_tests:
            lines.append(self.statistical_tests.summary())

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export as nested dictionary."""
        result: dict[str, Any] = {}

        if self.monte_carlo:
            mc = self.monte_carlo
            result["monte_carlo"] = {
                "n_simulations": mc.n_simulations,
                "equity_percentiles": mc.equity_percentiles,
                "drawdown_percentiles": mc.drawdown_percentiles,
                "risk_of_ruin": mc.risk_of_ruin,
                "ruin_threshold": mc.ruin_threshold,
                "expected_max_drawdown": mc.expected_max_drawdown,
                "expected_worst_equity": mc.expected_worst_equity,
                "original_final_equity": mc.original_final_equity,
                "original_max_drawdown": mc.original_max_drawdown,
            }

        if self.bootstrap:
            bs = self.bootstrap
            result["bootstrap"] = {
                "n_bootstrap": bs.n_bootstrap,
                "method": bs.method,
                "intervals": {
                    name: {
                        "point_estimate": ci.point_estimate,
                        "lower": ci.lower,
                        "upper": ci.upper,
                        "confidence_level": ci.confidence_level,
                    }
                    for name, ci in bs.intervals.items()
                },
            }

        if self.statistical_tests:
            st = self.statistical_tests
            result["statistical_tests"] = {
                "psr": st.psr,
                "psr_benchmark": st.psr_benchmark,
                "psr_is_significant": st.psr_is_significant,
                "min_track_record_length": st.min_track_record_length,
                "current_track_record": st.current_track_record,
                "track_record_sufficient": st.track_record_sufficient,
            }

        return result

    def plot(self) -> go.Figure:
        """Generate comprehensive validation summary plot."""
        from signalflow.analytic.stats.visualization import plot_validation_summary

        return plot_validation_summary(self)
