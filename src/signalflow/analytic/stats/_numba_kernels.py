"""Numba-accelerated kernels for statistical validation."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _fisher_yates_shuffle(arr: np.ndarray, seed: int) -> np.ndarray:
    """Fisher-Yates shuffle with specific seed.

    Args:
        arr: Array to shuffle (modified in-place)
        seed: Random seed

    Returns:
        Shuffled array (same as input, modified in-place)
    """
    np.random.seed(seed)
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


@njit(cache=True)
def _compute_equity_metrics(
    pnls: np.ndarray,
    initial_capital: float,
) -> tuple[float, float, int, int]:
    """Compute equity curve metrics from PnL sequence.

    Args:
        pnls: Array of trade PnLs
        initial_capital: Starting capital

    Returns:
        Tuple of (final_equity, max_drawdown, max_consecutive_losses, longest_dd_duration)
    """
    n_trades = len(pnls)

    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    consecutive_losses = 0
    max_consec_losses = 0
    dd_duration = 0
    max_dd_duration = 0

    for i in range(n_trades):
        equity += pnls[i]

        # Update drawdown
        if equity > peak:
            peak = equity
            dd_duration = 0
        else:
            if peak > 0:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)

        # Track consecutive losses
        if pnls[i] < 0:
            consecutive_losses += 1
            max_consec_losses = max(max_consec_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    return equity, max_dd, max_consec_losses, max_dd_duration


@njit(parallel=True, cache=True)
def simulate_equity_curves(
    pnls: np.ndarray,
    initial_capital: float,
    n_simulations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate equity curves by shuffling trade order.

    Uses Monte Carlo simulation to generate alternative equity curves
    by randomly reordering trade execution.

    Args:
        pnls: Array of trade PnLs from backtest
        initial_capital: Starting capital
        n_simulations: Number of simulations to run
        seed: Base random seed

    Returns:
        Tuple of:
        - final_equities: (n_simulations,) array of final equity values
        - max_drawdowns: (n_simulations,) array of maximum drawdowns
        - max_consecutive_losses: (n_simulations,) array of max consecutive losses
        - longest_dd_durations: (n_simulations,) array of longest drawdown durations
    """
    len(pnls)

    final_equities = np.empty(n_simulations, dtype=np.float64)
    max_drawdowns = np.empty(n_simulations, dtype=np.float64)
    max_consecutive_losses = np.empty(n_simulations, dtype=np.int32)
    longest_dd_durations = np.empty(n_simulations, dtype=np.int32)

    for sim in prange(n_simulations):
        # Create shuffled copy of PnLs
        shuffled = pnls.copy()
        _fisher_yates_shuffle(shuffled, seed + sim)

        # Compute metrics for this simulation
        final_eq, max_dd, max_cl, max_ddd = _compute_equity_metrics(shuffled, initial_capital)

        final_equities[sim] = final_eq
        max_drawdowns[sim] = max_dd
        max_consecutive_losses[sim] = max_cl
        longest_dd_durations[sim] = max_ddd

    return final_equities, max_drawdowns, max_consecutive_losses, longest_dd_durations


@njit(cache=True)
def compute_acceleration(theta_i: np.ndarray) -> float:
    """Compute acceleration factor for BCa bootstrap.

    The acceleration measures the rate of change of the standard error
    of the estimate with respect to the true parameter value.

    Args:
        theta_i: Jackknife estimates (leave-one-out)

    Returns:
        Acceleration factor 'a'
    """
    n = len(theta_i)
    theta_mean = np.mean(theta_i)

    num = 0.0
    denom = 0.0

    for i in range(n):
        diff = theta_mean - theta_i[i]
        num += diff**3
        denom += diff**2

    denom = denom**1.5

    if denom == 0:
        return 0.0

    return num / (6.0 * denom)


@njit(parallel=True, cache=True)
def bootstrap_sharpe_ratio(
    returns: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap resample Sharpe ratio calculation.

    Args:
        returns: Array of period returns
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Array of bootstrapped Sharpe ratios
    """
    n = len(returns)
    sharpes = np.empty(n_bootstrap, dtype=np.float64)

    for b in prange(n_bootstrap):
        np.random.seed(seed + b)

        # Sample with replacement
        sample = np.empty(n, dtype=np.float64)
        for i in range(n):
            idx = np.random.randint(0, n)
            sample[i] = returns[idx]

        mean_ret = np.mean(sample)
        std_ret = np.std(sample)

        if std_ret > 1e-10:
            sharpes[b] = mean_ret / std_ret
        else:
            sharpes[b] = 0.0

    return sharpes


@njit(parallel=True, cache=True)
def bootstrap_generic(
    data: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Generic bootstrap resampling (returns mean of each resample).

    Args:
        data: Array of values to bootstrap
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Array of bootstrapped means
    """
    n = len(data)
    results = np.empty(n_bootstrap, dtype=np.float64)

    for b in prange(n_bootstrap):
        np.random.seed(seed + b)

        total = 0.0
        for _i in range(n):
            idx = np.random.randint(0, n)
            total += data[idx]

        results[b] = total / n

    return results


@njit(cache=True)
def compute_sortino_ratio(returns: np.ndarray) -> float:
    """Compute Sortino ratio from returns.

    Args:
        returns: Array of period returns

    Returns:
        Sortino ratio
    """
    mean_ret = np.mean(returns)

    # Downside deviation (only negative returns)
    neg_returns_sq_sum = 0.0
    neg_count = 0

    for r in returns:
        if r < 0:
            neg_returns_sq_sum += r * r
            neg_count += 1

    if neg_count == 0:
        return 0.0 if mean_ret == 0 else np.inf

    downside_std = np.sqrt(neg_returns_sq_sum / neg_count)

    if downside_std < 1e-10:
        return 0.0 if mean_ret == 0 else np.inf

    return mean_ret / downside_std


@njit(cache=True)
def compute_calmar_ratio(returns: np.ndarray) -> float:
    """Compute Calmar ratio from returns.

    Args:
        returns: Array of period returns

    Returns:
        Calmar ratio (total return / max drawdown)
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # Compute cumulative returns
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0

    for r in returns:
        cumulative *= 1 + r
        if cumulative > peak:
            peak = cumulative
        if peak > 0:
            dd = (peak - cumulative) / peak
            if dd > max_dd:
                max_dd = dd

    total_return = cumulative - 1.0

    if max_dd < 1e-10:
        return 0.0 if total_return == 0 else np.inf

    return total_return / max_dd


@njit(cache=True)
def compute_profit_factor(pnls: np.ndarray) -> float:
    """Compute profit factor from PnLs.

    Args:
        pnls: Array of trade PnLs

    Returns:
        Profit factor (gross profit / gross loss)
    """
    gross_profit = 0.0
    gross_loss = 0.0

    for pnl in pnls:
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss -= pnl  # Make positive

    if gross_loss < 1e-10:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


@njit(cache=True)
def compute_win_rate(pnls: np.ndarray) -> float:
    """Compute win rate from PnLs.

    Args:
        pnls: Array of trade PnLs

    Returns:
        Win rate (0 to 1)
    """
    n = len(pnls)
    if n == 0:
        return 0.0

    wins = 0
    for pnl in pnls:
        if pnl > 0:
            wins += 1

    return wins / n
