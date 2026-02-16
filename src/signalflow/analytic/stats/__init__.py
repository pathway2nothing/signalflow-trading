"""SignalFlow Statistical Validation Module.

Provides Monte Carlo simulation, bootstrap confidence intervals,
and statistical significance tests for backtest validation.

Example:
    >>> import signalflow as sf
    >>> from signalflow.analytic.stats import MonteCarloSimulator, BootstrapValidator
    >>>
    >>> result = sf.Backtest("test").data(...).detector(...).run()
    >>>
    >>> # Quick validation via convenience functions
    >>> from signalflow.analytic.stats import monte_carlo, bootstrap, statistical_tests
    >>> mc = monte_carlo(result, n_simulations=10_000)
    >>> bs = bootstrap(result, method="bca")
    >>> tests = statistical_tests(result, sr_benchmark=0.5)
    >>>
    >>> # Or use validators directly for more control
    >>> mc_sim = MonteCarloSimulator(n_simulations=10_000, ruin_threshold=0.25)
    >>> mc_result = mc_sim.validate(result)
    >>> print(mc_result.summary())
"""

from signalflow.analytic.stats.bootstrap import BootstrapValidator, bootstrap
from signalflow.analytic.stats.monte_carlo import MonteCarloSimulator, monte_carlo
from signalflow.analytic.stats.results import (
    BootstrapResult,
    ConfidenceInterval,
    MonteCarloResult,
    StatisticalTestResult,
    ValidationResult,
)
from signalflow.analytic.stats.statistical_tests import StatisticalTestsValidator, statistical_tests
from signalflow.analytic.stats.visualization import (
    plot_bootstrap,
    plot_monte_carlo,
    plot_validation_summary,
)

__all__ = [
    # Validators
    "MonteCarloSimulator",
    "BootstrapValidator",
    "StatisticalTestsValidator",
    # Convenience functions
    "monte_carlo",
    "bootstrap",
    "statistical_tests",
    # Results
    "MonteCarloResult",
    "BootstrapResult",
    "StatisticalTestResult",
    "ValidationResult",
    "ConfidenceInterval",
    # Visualization
    "plot_monte_carlo",
    "plot_bootstrap",
    "plot_validation_summary",
]
