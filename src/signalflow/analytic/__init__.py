from signalflow.analytic.base import SignalMetric, StrategyMetric
from signalflow.analytic.compare import ComparisonResult, compare_results
from signalflow.analytic.stats import (
    BootstrapResult,
    BootstrapValidator,
    ConfidenceInterval,
    MonteCarloResult,
    MonteCarloSimulator,
    StatisticalTestResult,
    StatisticalTestsValidator,
    ValidationResult,
    bootstrap,
    monte_carlo,
    plot_bootstrap,
    plot_monte_carlo,
    plot_validation_summary,
    statistical_tests,
)

__all__ = [
    "BootstrapResult",
    "BootstrapValidator",
    # Comparison
    "ComparisonResult",
    "ConfidenceInterval",
    # Stats - Results
    "MonteCarloResult",
    # Stats - Validators
    "MonteCarloSimulator",
    # Base
    "SignalMetric",
    "StatisticalTestResult",
    "StatisticalTestsValidator",
    "StrategyMetric",
    "ValidationResult",
    "bootstrap",
    "compare_results",
    # Stats - Convenience functions
    "monte_carlo",
    "plot_bootstrap",
    # Stats - Visualization
    "plot_monte_carlo",
    "plot_validation_summary",
    "statistical_tests",
]
