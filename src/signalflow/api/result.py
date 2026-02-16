"""
BacktestResult - Rich container for backtest results.

Wraps existing analytics (StrategyMainResult, StrategyPairResult) from
signalflow.analytic.strategy for visualization and metrics computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

import polars as pl

from signalflow.core import StrategyState, RawData, Signals, default_registry, SfComponentType

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from signalflow.analytic.stats.results import (
        BootstrapResult,
        MonteCarloResult,
        StatisticalTestResult,
        ValidationResult,
    )


# =============================================================================
# Type Definitions
# =============================================================================


@runtime_checkable
class TradeProtocol(Protocol):
    """Protocol for trade objects.

    Trades can come from different sources (runner, external systems),
    but must have these minimal attributes for metrics computation.
    """

    @property
    def pnl(self) -> float | None:
        """Profit/loss from this trade."""
        ...

    @property
    def pair(self) -> str:
        """Trading pair."""
        ...


# Type alias for trades - can be TradeProtocol or dict
TradeType = TradeProtocol | dict[str, Any]


# =============================================================================
# BacktestResult
# =============================================================================


@dataclass
class BacktestResult:
    """
    Container for backtest results with convenient access to existing analytics.

    Wraps StrategyMainResult and StrategyPairResult from signalflow.analytic.strategy
    for visualization and metrics computation. Uses registry to discover and
    instantiate metric components.

    Attributes:
        state: Final strategy state after backtest
        trades: List of executed trades
        signals: Signals used in backtest
        raw: Raw market data used
        config: Backtest configuration dict
        metrics_df: Time-series metrics DataFrame (optional)

    Example:
        >>> result = sf.Backtest("test").data(...).detector(...).run()
        >>> print(result.summary())
        >>> result.plot()
        >>> print(result.metrics)

    Jupyter Support:
        BacktestResult renders as an HTML table in Jupyter notebooks,
        showing key metrics at a glance.
    """

    state: StrategyState
    trades: Sequence[TradeType]
    signals: Signals | None
    raw: RawData
    config: dict[str, Any]
    metrics_df: pl.DataFrame | None = None

    # Cached analytics (created via registry on first access)
    _main_result: Any | None = field(default=None, repr=False)
    _metrics_cache: dict[str, float] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize analytics from registry."""
        try:
            main_cls = default_registry.get(SfComponentType.STRATEGY_METRIC, "result_main")
            self._main_result = main_cls()
        except KeyError:
            pass

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def n_trades(self) -> int:
        """Number of trades executed."""
        return len(self.trades) if self.trades else 0

    @property
    def final_capital(self) -> float:
        """Final capital after backtest."""
        return float(getattr(self.state, "capital", 0.0))

    @property
    def initial_capital(self) -> float:
        """Initial capital from config."""
        return float(self.config.get("capital", 10_000.0))

    @property
    def total_return(self) -> float:
        """Total return as decimal (e.g., 0.15 = 15%)."""
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        return self.total_return * 100

    @property
    def win_rate(self) -> float:
        """Win rate as decimal (e.g., 0.6 = 60%)."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if self._get_trade_pnl(t) > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses."""
        if not self.trades:
            return 0.0

        gross_profit = sum(self._get_trade_pnl(t) for t in self.trades if self._get_trade_pnl(t) > 0)
        gross_loss = abs(sum(self._get_trade_pnl(t) for t in self.trades if self._get_trade_pnl(t) < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    # =========================================================================
    # Metrics (from existing analytics via registry)
    # =========================================================================

    @property
    def metrics(self) -> dict[str, float]:
        """
        Compute all metrics using registered STRATEGY_METRIC components.

        Returns cached result on subsequent calls.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        results: dict[str, float] = {}
        prices = self._get_last_prices()

        # Add basic metrics
        results["n_trades"] = float(self.n_trades)
        results["win_rate"] = self.win_rate
        results["total_return"] = self.total_return
        results["profit_factor"] = self.profit_factor
        results["initial_capital"] = self.initial_capital
        results["final_capital"] = self.final_capital

        # Get metrics from registry
        for name in default_registry.list(SfComponentType.STRATEGY_METRIC):
            # Skip visualization-only metrics
            if name in ("result_main", "result_pair"):
                continue

            try:
                metric_cls = default_registry.get(SfComponentType.STRATEGY_METRIC, name)
                metric = metric_cls()
                computed = metric.compute(self.state, prices)
                if computed and isinstance(computed, dict):
                    results.update(computed)
            except Exception:
                pass

        self._metrics_cache = results
        return results

    # =========================================================================
    # Visualization (delegates to existing analytics)
    # =========================================================================

    def plot(self) -> list[go.Figure] | None:
        """
        Plot strategy results using StrategyMainResult.

        Returns list of Plotly figures or None if plotting unavailable.
        """
        if self._main_result is None:
            return None

        results_dict = self._build_results_dict()
        return self._main_result.plot(
            results=results_dict,
            state=self.state,
            raw_data=self.raw,
        )

    def plot_pair(self, pair: str) -> list[go.Figure] | None:
        """
        Plot pair-level results using StrategyPairResult.

        Args:
            pair: Trading pair to plot (e.g., "BTCUSDT")

        Returns:
            List of Plotly figures or None if plotting unavailable.
        """
        try:
            pair_cls = default_registry.get(SfComponentType.STRATEGY_METRIC, "result_pair")
            pair_result = pair_cls(pairs=[pair])

            results_dict = self._build_results_dict()
            return pair_result.plot(
                results=results_dict,
                state=self.state,
                raw_data=self.raw,
            )
        except KeyError:
            return None

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> str:
        """
        Return formatted summary string.

        Includes key metrics in a readable format.
        """
        m = self.metrics

        # Get additional metrics if available
        max_dd = m.get("max_drawdown", 0)
        sharpe = m.get("sharpe_ratio", 0)

        lines = [
            "",
            "=" * 50,
            "           BACKTEST SUMMARY",
            "=" * 50,
            f"  Trades:          {self.n_trades:>10}",
            f"  Win Rate:        {self.win_rate:>10.1%}",
            f"  Profit Factor:   {self.profit_factor:>10.2f}",
            "-" * 50,
            f"  Initial Capital: ${self.initial_capital:>12,.2f}",
            f"  Final Capital:   ${self.final_capital:>12,.2f}",
            f"  Total Return:    {self.total_return:>+10.1%}",
            "-" * 50,
        ]

        if max_dd != 0:
            lines.append(f"  Max Drawdown:    {max_dd:>10.1%}")
        if sharpe != 0:
            lines.append(f"  Sharpe Ratio:    {sharpe:>10.2f}")

        lines.append("=" * 50)
        lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"BacktestResult(trades={self.n_trades}, return={self.total_return:+.1%}, win_rate={self.win_rate:.1%})"

    # =========================================================================
    # Jupyter Support
    # =========================================================================

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        m = self.metrics
        max_dd = m.get("max_drawdown", 0)
        sharpe = m.get("sharpe_ratio", 0)

        # Determine color for return
        return_color = "#22c55e" if self.total_return >= 0 else "#ef4444"

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 500px;">
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 20px; border-radius: 12px 12px 0 0;">
                <h3 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500; opacity: 0.8;">BACKTEST RESULT</h3>
                <div style="font-size: 32px; font-weight: 700; color: {return_color};">
                    {self.total_return:+.2%}
                </div>
                <div style="font-size: 12px; opacity: 0.7; margin-top: 4px;">
                    ${self.initial_capital:,.0f} → ${self.final_capital:,.0f}
                </div>
            </div>
            <div style="background: #f8fafc; padding: 16px; border-radius: 0 0 12px 12px; border: 1px solid #e2e8f0; border-top: none;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Trades</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{self.n_trades}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Win Rate</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{self.win_rate:.1%}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Profit Factor</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{self.profit_factor:.2f}</td>
                    </tr>
        """

        if max_dd != 0:
            html += f"""
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Max Drawdown</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600; color: #ef4444;">{max_dd:.1%}</td>
                    </tr>
            """

        if sharpe != 0:
            html += f"""
                    <tr>
                        <td style="padding: 8px 0; color: #64748b;">Sharpe Ratio</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">{sharpe:.2f}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </div>
        """
        return html

    # =========================================================================
    # Export
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Export results as dictionary."""
        return {
            "metrics": self.metrics,
            "n_trades": self.n_trades,
            "trades": [self._trade_to_dict(t) for t in self.trades] if self.trades else [],
            "config": self.config,
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Export fully JSON-serializable result with equity curve.

        All datetime values are converted to ISO 8601 strings.
        Polars types are converted to Python native types.

        Returns:
            Dict with keys: ``metrics``, ``n_trades``, ``trades``,
            ``equity_curve``, ``config``.
        """
        return {
            "metrics": self.metrics,
            "n_trades": self.n_trades,
            "trades": [self._trade_to_json_safe(t) for t in self.trades] if self.trades else [],
            "equity_curve": self._get_equity_curve(),
            "config": self._json_safe(self.config),
        }

    def _get_equity_curve(self) -> list[dict[str, Any]]:
        """Extract equity curve from metrics_df as JSON-safe list of dicts."""
        if self.metrics_df is None or self.metrics_df.height == 0:
            return []
        rows = self.metrics_df.to_dicts()
        return [self._json_safe(row) for row in rows]

    def _trade_to_json_safe(self, trade: TradeType) -> dict[str, Any]:
        """Convert trade to JSON-safe dict (datetimes → ISO strings)."""
        d = self._trade_to_dict(trade)
        return self._json_safe(d)

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively convert non-JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: BacktestResult._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BacktestResult._json_safe(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None  # NaN / Inf → null
        if hasattr(obj, "item"):
            return obj.item()  # numpy/polars scalar → Python native
        return obj

    def to_dataframe(self) -> pl.DataFrame:
        """Export trades as Polars DataFrame.

        Returns:
            DataFrame with trade details (pair, pnl, etc.)
        """
        if not self.trades:
            return pl.DataFrame()

        trade_dicts = [self._trade_to_dict(t) for t in self.trades]
        return pl.DataFrame(trade_dicts)

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _build_results_dict(self) -> dict[str, Any]:
        """Build results dict for analytics components."""
        return {
            "metrics_df": self.metrics_df,
            "initial_capital": self.initial_capital,
            "final_return": self.total_return,
            "max_drawdown": self.metrics.get("max_drawdown", 0),
        }

    def _get_last_prices(self) -> dict[str, float]:
        """Get last prices from raw data for metric computation."""
        prices: dict[str, float] = {}
        try:
            # Try different data access patterns
            spot = None
            if hasattr(self.raw, "spot"):
                accessor = self.raw.spot
                if hasattr(accessor, "to_polars"):
                    spot = accessor.to_polars()
                elif isinstance(accessor, pl.DataFrame):
                    spot = accessor

            if spot is not None and "pair" in spot.columns and "close" in spot.columns:
                for pair in spot["pair"].unique().to_list():
                    last = spot.filter(pl.col("pair") == pair).tail(1)
                    if last.height > 0:
                        prices[pair] = float(last["close"][0])
        except Exception:
            pass
        return prices

    def _get_trade_pnl(self, trade: TradeType) -> float:
        """Extract PnL from trade object."""
        if hasattr(trade, "pnl"):
            return float(trade.pnl or 0)
        if hasattr(trade, "realized_pnl"):
            return float(trade.realized_pnl or 0)  # type: ignore[union-attr]
        if isinstance(trade, dict):
            return float(trade.get("pnl", 0) or trade.get("realized_pnl", 0))
        return 0.0

    def _trade_to_dict(self, trade: TradeType) -> dict[str, Any]:
        """Convert trade object to dictionary."""
        if isinstance(trade, dict):
            return trade
        if hasattr(trade, "__dict__"):
            return {k: v for k, v in trade.__dict__.items() if not k.startswith("_")}
        return {"trade": str(trade)}

    # =========================================================================
    # Statistical Validation
    # =========================================================================

    def monte_carlo(
        self,
        n_simulations: int = 10_000,
        ruin_threshold: float = 0.20,
        random_seed: int | None = None,
        confidence_levels: tuple[float, ...] = (0.05, 0.50, 0.95),
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on trade sequence.

        Shuffles trade execution order to estimate distribution of outcomes
        under different trade sequences. Useful for assessing strategy
        robustness and estimating risk metrics.

        Args:
            n_simulations: Number of simulations to run (default: 10,000)
            ruin_threshold: Max drawdown threshold for risk of ruin (default: 20%)
            random_seed: Random seed for reproducibility (None for random)
            confidence_levels: Percentile levels to compute (default: 5%, 50%, 95%)

        Returns:
            MonteCarloResult with simulation distributions and risk metrics

        Example:
            >>> result = backtest.run()
            >>> mc = result.monte_carlo(n_simulations=10_000)
            >>> print(f"Risk of Ruin: {mc.risk_of_ruin:.1%}")
            >>> print(f"5th percentile equity: ${mc.equity_percentiles[0.05]:,.2f}")
            >>> mc.plot()
        """
        from signalflow.analytic.stats import MonteCarloSimulator

        simulator = MonteCarloSimulator(
            n_simulations=n_simulations,
            ruin_threshold=ruin_threshold,
            random_seed=random_seed,
            confidence_levels=confidence_levels,
        )
        return simulator.validate(self)

    def bootstrap(
        self,
        n_bootstrap: int = 5_000,
        method: str = "bca",
        confidence_level: float = 0.95,
        metrics: tuple[str, ...] | None = None,
        block_size: int | None = None,
        random_seed: int | None = None,
    ) -> BootstrapResult:
        """Compute bootstrap confidence intervals for key metrics.

        Estimates confidence intervals for performance metrics using
        bootstrap resampling. Supports BCa (bias-corrected accelerated),
        percentile, and block bootstrap methods.

        Args:
            n_bootstrap: Number of bootstrap resamples (default: 5,000)
            method: Bootstrap method - "bca", "percentile", or "block"
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            metrics: Metrics to bootstrap (default: sharpe, sortino, calmar, profit_factor, win_rate)
            block_size: Block size for block bootstrap (auto if None)
            random_seed: Random seed for reproducibility

        Returns:
            BootstrapResult with confidence intervals for each metric

        Example:
            >>> result = backtest.run()
            >>> bs = result.bootstrap(method="bca", confidence_level=0.95)
            >>> sr_ci = bs.intervals["sharpe_ratio"]
            >>> print(f"Sharpe: {sr_ci.point_estimate:.2f} ({sr_ci.lower:.2f}, {sr_ci.upper:.2f})")
            >>> print(f"Significant vs 0: {bs.is_significant('sharpe_ratio', 0)}")
        """
        from signalflow.analytic.stats import BootstrapValidator

        if metrics is None:
            metrics = (
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "profit_factor",
                "win_rate",
            )

        validator = BootstrapValidator(
            n_bootstrap=n_bootstrap,
            method=method,  # type: ignore[arg-type]
            block_size=block_size,
            confidence_level=confidence_level,
            random_seed=random_seed,
            metrics=metrics,
        )
        return validator.validate(self)

    def statistical_tests(
        self,
        sr_benchmark: float = 0.0,
        confidence_level: float = 0.95,
    ) -> StatisticalTestResult:
        """Run statistical significance tests on backtest results.

        Computes:
        - Probabilistic Sharpe Ratio (PSR): probability SR > benchmark
        - Minimum Track Record Length (MinTRL): trades needed for significance

        Based on Bailey & Lopez de Prado (2012).

        Args:
            sr_benchmark: Benchmark Sharpe ratio to compare against (default: 0)
            confidence_level: Required confidence level (default: 0.95)

        Returns:
            StatisticalTestResult with PSR and MinTRL values

        Example:
            >>> result = backtest.run()
            >>> tests = result.statistical_tests(sr_benchmark=0.5)
            >>> print(f"PSR: {tests.psr:.1%}")
            >>> print(f"Significant: {tests.psr_is_significant}")
            >>> print(f"Min trades needed: {tests.min_track_record_length}")
        """
        from signalflow.analytic.stats import StatisticalTestsValidator

        validator = StatisticalTestsValidator(
            sr_benchmark=sr_benchmark,
            confidence_level=confidence_level,
        )
        return validator.validate(self)

    def validate(
        self,
        monte_carlo: bool = True,
        bootstrap: bool = True,
        statistical_tests: bool = True,
        mc_simulations: int = 10_000,
        bs_resamples: int = 5_000,
        confidence_level: float = 0.95,
        ruin_threshold: float = 0.20,
    ) -> ValidationResult:
        """Run comprehensive statistical validation.

        Combines Monte Carlo simulation, bootstrap confidence intervals,
        and statistical significance tests into a single analysis.

        Args:
            monte_carlo: Run Monte Carlo simulation (default: True)
            bootstrap: Run bootstrap confidence intervals (default: True)
            statistical_tests: Run statistical tests - PSR, MinTRL (default: True)
            mc_simulations: Number of Monte Carlo simulations
            bs_resamples: Number of bootstrap resamples
            confidence_level: Confidence level for all tests
            ruin_threshold: Max drawdown threshold for risk of ruin

        Returns:
            ValidationResult combining all analyses

        Example:
            >>> result = backtest.run()
            >>> validation = result.validate()
            >>> print(validation.summary())
            >>> validation.plot()
        """
        from signalflow.analytic.stats import ValidationResult

        mc_result = (
            self.monte_carlo(
                n_simulations=mc_simulations,
                ruin_threshold=ruin_threshold,
            )
            if monte_carlo
            else None
        )

        bs_result = (
            self.bootstrap(
                n_bootstrap=bs_resamples,
                confidence_level=confidence_level,
            )
            if bootstrap
            else None
        )

        st_result = (
            self.statistical_tests(
                confidence_level=confidence_level,
            )
            if statistical_tests
            else None
        )

        return ValidationResult(
            monte_carlo=mc_result,
            bootstrap=bs_result,
            statistical_tests=st_result,
        )
