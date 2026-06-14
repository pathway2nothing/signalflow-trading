---
title: Statistical Analysis
description: Monte Carlo simulation, Bootstrap CI, and significance tests with Numba acceleration
---

# Statistical Analysis

SignalFlow provides three statistical validation frameworks for assessing
strategy robustness. All numerical kernels are **Numba JIT-compiled** for
high performance.

---

## Quick Start

```python
from signalflow.analytic.stats import monte_carlo, bootstrap, statistical_tests

result = sf.Backtest("test").data(raw=data).detector("sma_cross").run()

# Monte Carlo - trade order shuffling
mc = monte_carlo(result, n_simulations=10_000)
print(f"Risk of Ruin: {mc.risk_of_ruin:.1%}")
print(f"Expected Max Drawdown: {mc.expected_max_drawdown:.2%}")

# Bootstrap - confidence intervals
bs = bootstrap(result, method="bca", confidence_level=0.95)
print(bs.intervals["sharpe_ratio"])

# Statistical tests - PSR & MinTRL
tests = statistical_tests(result, sr_benchmark=0.5)
print(f"PSR: {tests.psr:.2%}")
print(f"Min trades needed: {tests.min_track_record_length}")
```

---

## Monte Carlo Simulation

Randomizes trade execution order across thousands of simulations to estimate
the distribution of outcomes. Answers: *"How lucky/unlucky was this specific
trade sequence?"*

```python
from signalflow.analytic.stats import MonteCarloSimulator

mc = MonteCarloSimulator(
    n_simulations=10_000,
    ruin_threshold=0.20,        # 20% drawdown = ruin
    random_seed=42,
    confidence_levels=(0.05, 0.50, 0.95),
)
result = mc.validate(backtest_result)
```

### MonteCarloResult

| Attribute | Description |
|-----------|-------------|
| `final_equity_dist` | Distribution of final equity values |
| `max_drawdown_dist` | Distribution of maximum drawdowns |
| `max_consecutive_losses_dist` | Distribution of losing streak lengths |
| `equity_percentiles` | Equity at 5th, 50th, 95th percentiles |
| `drawdown_percentiles` | Drawdown at 5th, 50th, 95th percentiles |
| `risk_of_ruin` | P(max drawdown > threshold) |
| `expected_max_drawdown` | Mean of drawdown distribution |
| `expected_worst_equity` | 5th percentile of equity |

```python
mc_result = monte_carlo(result, n_simulations=10_000)

# Visualize
mc_result.plot()    # 3 Plotly figures: equity fan, drawdown dist, ruin curve

# Text summary
print(mc_result.summary())
```

---

## Bootstrap Confidence Intervals

Estimates uncertainty of performance metrics through resampling. Three methods:

| Method | Use Case |
|--------|----------|
| **BCa** (bias-corrected accelerated) | General metrics, adjusts for bias and skewness |
| **Percentile** | Simple intervals, no correction |
| **Block** | Time series with autocorrelation |

```python
from signalflow.analytic.stats import BootstrapValidator

bs = BootstrapValidator(
    n_bootstrap=5_000,
    method="bca",               # "bca", "percentile", "block"
    confidence_level=0.95,
    block_size=None,            # Auto for block bootstrap
    metrics=(
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor",
        "win_rate",
    ),
)
result = bs.validate(backtest_result)
```

### Available Metrics

| Metric | Formula |
|--------|---------|
| `sharpe_ratio` | Mean return / Std of returns |
| `sortino_ratio` | Mean return / Downside std |
| `calmar_ratio` | Total return / Max drawdown |
| `profit_factor` | Gross profit / Gross loss |
| `win_rate` | Winning trades / Total trades |

### BootstrapResult

```python
bs_result = bootstrap(result, method="bca")

# Access confidence intervals
ci = bs_result.intervals["sharpe_ratio"]
print(f"Sharpe: {ci.point_estimate:.2f} [{ci.lower:.2f}, {ci.upper:.2f}]")

# Full bootstrap distributions
dist = bs_result.distributions["sharpe_ratio"]  # np.ndarray

# Visualize
bs_result.plot()    # Forest plot with CIs
```

---

## Statistical Significance Tests

Two tests from Bailey & Lopez de Prado (2012):

### Probabilistic Sharpe Ratio (PSR)

*"What is the probability that the true Sharpe ratio exceeds a benchmark?"*

Accounts for skewness and kurtosis of returns - unlike the naive Sharpe ratio
which assumes normality.

```python
from signalflow.analytic.stats import StatisticalTestsValidator

tests = StatisticalTestsValidator(
    sr_benchmark=0.0,           # Benchmark to beat
    confidence_level=0.95,
)
result = tests.validate(backtest_result)
```

| Attribute | Description |
|-----------|-------------|
| `psr` | Probability that true SR > benchmark (0–1) |
| `psr_is_significant` | Whether PSR > confidence_level |

### Minimum Track Record Length (MinTRL)

*"How many observations do we need for the Sharpe ratio to be statistically
significant?"*

| Attribute | Description |
|-----------|-------------|
| `min_track_record_length` | Minimum trades needed for significance |
| `current_track_record` | Current number of observations |
| `track_record_sufficient` | Whether current data is enough |

```python
tests_result = statistical_tests(result, sr_benchmark=0.5)

if tests_result.track_record_sufficient:
    print("Sufficient data for significance")
else:
    needed = tests_result.min_track_record_length - tests_result.current_track_record
    print(f"Need {needed} more observations")
```

---

## Combined Validation

```python
from signalflow.analytic.stats import ValidationResult

combined = ValidationResult(
    monte_carlo=mc_result,
    bootstrap=bs_result,
    statistical_tests=tests_result,
)

# Dashboard with all results
combined.plot()         # 2x2 Plotly dashboard
print(combined.summary())
```

---

## Numba Acceleration

All compute-intensive kernels use `@njit(cache=True)` with parallel support:

| Kernel | Parallelization |
|--------|----------------|
| `simulate_equity_curves()` | `prange` - parallel simulations |
| `bootstrap_sharpe_ratio()` | `prange` - parallel resamples |
| `bootstrap_generic()` | `prange` - parallel resamples |
| `compute_acceleration()` | Single-threaded (jackknife) |
| Metric functions | Single-threaded |

Kernels are JIT-compiled on first run and cached to disk. Subsequent calls
use the cached machine code.

**Typical runtimes:**

- 10,000 Monte Carlo simulations: ~50–200ms
- 5,000 bootstrap resamples: ~100–300ms
- Statistical tests: ~10–50ms

---

## Imports

```python
# Convenience functions
from signalflow.analytic.stats import monte_carlo, bootstrap, statistical_tests

# Class-based API
from signalflow.analytic.stats import (
    MonteCarloSimulator,
    BootstrapValidator,
    StatisticalTestsValidator,
    ValidationResult,
)

# Result types
from signalflow.analytic.stats import (
    MonteCarloResult,
    BootstrapResult,
    StatisticalTestResult,
    ConfidenceInterval,
)
```
