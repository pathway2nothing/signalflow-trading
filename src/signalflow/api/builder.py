"""
BacktestBuilder - Fluent builder for backtest configuration.

Uses SignalFlowRegistry for component discovery and creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self

from signalflow.core import (
    default_registry,
    SfComponentType,
    RawData,
    Signals,
)
from signalflow.api.exceptions import (
    DetectorNotFoundError,
    MissingDataError,
    MissingDetectorError,
    InvalidParameterError,
)

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.api.result import BacktestResult


@dataclass
class BacktestBuilder:
    """
    Fluent builder for backtest configuration.

    Uses SignalFlowRegistry for component discovery and creation.
    Provides a chainable API for constructing backtests with clear,
    readable configuration.

    Example:
        >>> import signalflow as sf
        >>>
        >>> result = (
        ...     sf.Backtest("my_strategy")
        ...     .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        ...     .detector("example/sma_cross", fast_period=20, slow_period=50)
        ...     .entry(size_pct=0.1, max_positions=5)
        ...     .exit(tp=0.03, sl=0.015, trailing=0.02)
        ...     .capital(50_000)
        ...     .run()
        ... )

    Attributes:
        strategy_id: Unique identifier for the strategy
    """

    strategy_id: str = "backtest"

    # Internal state
    _raw: RawData | None = field(default=None, repr=False)
    _detector: SignalDetector | None = field(default=None, repr=False)
    _signals: Signals | None = field(default=None, repr=False)
    _entry_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _exit_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _capital: float = 10_000.0
    _fee: float = 0.001
    _show_progress: bool = True
    _data_params: dict[str, Any] | None = field(default=None, repr=False)

    # =========================================================================
    # Data Configuration
    # =========================================================================

    def data(
        self,
        raw: RawData | None = None,
        *,
        exchange: str | None = None,
        pairs: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        timeframe: str = "1m",
        data_type: str = "perpetual",
    ) -> Self:
        """
        Configure data source.

        Either pass pre-loaded RawData or specify exchange parameters
        for lazy loading.

        Args:
            raw: Pre-loaded RawData instance
            exchange: Exchange name ("binance", "okx", "bybit")
            pairs: List of trading pairs
            start: Start date (ISO string or datetime)
            end: End date (default: now)
            timeframe: Candle timeframe (default: "1h")
            data_type: Data type ("spot", "futures", "perpetual")

        Returns:
            Self for method chaining

        Examples:
            >>> .data(raw=my_raw_data)
            >>> .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        """
        if raw is not None:
            self._raw = raw
        else:
            self._data_params = {
                "exchange": exchange,
                "pairs": pairs,
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "data_type": data_type,
            }
        return self

    # =========================================================================
    # Detector Configuration
    # =========================================================================

    def detector(
        self,
        detector: SignalDetector | str,
        **kwargs: Any,
    ) -> Self:
        """
        Set signal detector.

        Accepts either a detector instance or a registry name string.

        Args:
            detector: SignalDetector instance OR registry name (e.g., "example/sma_cross")
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining

        Examples:
            >>> .detector(MyDetector(param=1))                     # Instance
            >>> .detector("example/sma_cross", fast_period=20)     # Registry name
        """
        if isinstance(detector, str):
            # Create from registry with helpful error
            try:
                self._detector = default_registry.create(
                    SfComponentType.DETECTOR,
                    detector,
                    **kwargs,
                )
            except KeyError:
                available = default_registry.list(SfComponentType.DETECTOR)
                raise DetectorNotFoundError(detector, available) from None
        else:
            self._detector = detector
        return self

    def signals(self, signals: Signals) -> Self:
        """
        Use pre-computed signals (skip detection).

        Args:
            signals: Pre-computed Signals instance

        Returns:
            Self for method chaining
        """
        self._signals = signals
        return self

    # =========================================================================
    # Entry Configuration
    # =========================================================================

    def entry(
        self,
        *,
        rule: str | None = None,
        size: float | None = None,
        size_pct: float | None = None,
        max_positions: int = 10,
        max_per_pair: int = 1,
        **kwargs: Any,
    ) -> Self:
        """
        Configure entry rules.

        Args:
            rule: Registry name for custom entry rule (e.g., "signal")
            size: Fixed position size in quote currency
            size_pct: Position size as % of capital (overrides size)
            max_positions: Maximum total concurrent positions
            max_per_pair: Maximum positions per trading pair
            **kwargs: Additional params for custom rule

        Returns:
            Self for method chaining

        Examples:
            >>> .entry(size=100, max_positions=5)
            >>> .entry(size_pct=0.1, max_per_pair=2)
        """
        self._entry_config = {
            "rule": rule,
            "size": size,
            "size_pct": size_pct,
            "max_positions": max_positions,
            "max_per_pair": max_per_pair,
            **kwargs,
        }
        return self

    # =========================================================================
    # Exit Configuration
    # =========================================================================

    def exit(
        self,
        *,
        rule: str | None = None,
        tp: float | None = None,
        sl: float | None = None,
        trailing: float | None = None,
        time_limit: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Configure exit rules.

        Args:
            rule: Registry name for custom exit rule
            tp: Take profit percentage (e.g., 0.03 = 3%)
            sl: Stop loss percentage (e.g., 0.015 = 1.5%)
            trailing: Trailing stop percentage
            time_limit: Maximum bars to hold position
            **kwargs: Additional params for custom rule

        Returns:
            Self for method chaining

        Examples:
            >>> .exit(tp=0.03, sl=0.015)
            >>> .exit(tp=0.05, sl=0.02, trailing=0.03)
        """
        self._exit_config = {
            "rule": rule,
            "tp": tp,
            "sl": sl,
            "trailing": trailing,
            "time_limit": time_limit,
            **kwargs,
        }
        return self

    # =========================================================================
    # Other Configuration
    # =========================================================================

    def capital(self, amount: float) -> Self:
        """
        Set initial capital.

        Args:
            amount: Initial capital in quote currency

        Returns:
            Self for method chaining

        Raises:
            InvalidParameterError: If amount is not positive
        """
        if amount <= 0:
            raise InvalidParameterError(
                "capital",
                amount,
                "Capital must be a positive number.",
                hint="Use a value like 10_000 or 50_000.0",
            )
        self._capital = amount
        return self

    def fee(self, rate: float) -> Self:
        """
        Set trading fee rate.

        Args:
            rate: Fee rate as decimal (e.g., 0.001 = 0.1%)

        Returns:
            Self for method chaining
        """
        self._fee = rate
        return self

    def progress(self, show: bool = True) -> Self:
        """
        Enable/disable progress bar during backtest.

        Args:
            show: Whether to show progress bar

        Returns:
            Self for method chaining
        """
        self._show_progress = show
        return self

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self) -> BacktestResult:
        """
        Execute backtest and return results.

        Resolves all configuration, creates components from registry,
        runs the backtest, and wraps results in BacktestResult.

        Returns:
            BacktestResult with trades, metrics, and analytics

        Raises:
            ValueError: If data or detector not configured
        """
        from signalflow.api.result import BacktestResult

        # 1. Resolve data
        raw = self._resolve_data()

        # 2. Resolve signals
        signals = self._resolve_signals(raw)

        # 3. Build components from registry
        entry_rules = self._build_entry_rules()
        exit_rules = self._build_exit_rules()
        broker = self._build_broker()

        # 4. Create and run runner
        runner = self._build_runner(broker, entry_rules, exit_rules)
        state = runner.run(raw_data=raw, signals=signals)

        # 5. Wrap in BacktestResult
        return BacktestResult(
            state=state,
            trades=getattr(runner, "trades", []),
            signals=signals,
            raw=raw,
            config={
                "capital": self._capital,
                "fee": self._fee,
                **self._entry_config,
                **self._exit_config,
            },
            metrics_df=getattr(runner, "metrics_df", None),
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of error/warning messages
        """
        issues: list[str] = []

        # Check data
        if self._raw is None and not self._data_params:
            issues.append("ERROR: No data source configured. Use .data()")
        elif self._data_params:
            if not self._data_params.get("pairs"):
                issues.append("ERROR: No pairs specified in .data()")
            if not self._data_params.get("start"):
                issues.append("ERROR: No start date specified in .data()")

        # Check detector/signals
        if self._detector is None and self._signals is None:
            issues.append("ERROR: No detector or signals configured. Use .detector() or .signals()")

        # Check registry availability
        try:
            default_registry.get(SfComponentType.STRATEGY_RUNNER, "backtest")
        except KeyError:
            issues.append("ERROR: BacktestRunner not found in registry")

        # Validate TP/SL ratio
        tp = self._exit_config.get("tp")
        sl = self._exit_config.get("sl")
        if tp and sl and tp < sl:
            issues.append(f"WARNING: TP ({tp:.1%}) < SL ({sl:.1%}), risk/reward < 1")

        # Validate capital
        if self._capital <= 0:
            issues.append("ERROR: Capital must be positive")

        return issues

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _resolve_data(self) -> RawData:
        """Load data from params or return pre-loaded."""
        if self._raw is not None:
            return self._raw

        if not self._data_params:
            raise MissingDataError()

        from signalflow.api.shortcuts import load

        # Filter out None values
        params = {k: v for k, v in self._data_params.items() if v is not None}
        return load(**params)

    def _resolve_signals(self, raw: RawData) -> Signals:
        """Detect signals or return pre-computed."""
        if self._signals is not None:
            return self._signals

        if self._detector is None:
            raise MissingDetectorError()

        return self._detector.run(raw.view())

    def _build_entry_rules(self) -> list[Any]:
        """Build entry rules from config using registry."""
        rule_name = self._entry_config.get("rule", "signal")

        try:
            rule_cls = default_registry.get(SfComponentType.STRATEGY_ENTRY_RULE, rule_name)
        except KeyError:
            # Fallback to SignalEntryRule
            from signalflow.strategy.component.entry.signal import SignalEntryRule
            rule_cls = SignalEntryRule

        # Calculate position size
        size = self._entry_config.get("size", 100.0)
        size_pct = self._entry_config.get("size_pct")
        if size_pct:
            size = self._capital * size_pct

        rule = rule_cls(
            base_position_size=size,
            max_positions_per_pair=self._entry_config.get("max_per_pair", 1),
            max_total_positions=self._entry_config.get("max_positions", 10),
        )
        return [rule]

    def _build_exit_rules(self) -> list[Any]:
        """Build exit rules from config using registry."""
        rules: list[Any] = []

        # TP/SL rule
        tp = self._exit_config.get("tp")
        sl = self._exit_config.get("sl")
        if tp or sl:
            try:
                tpsl_cls = default_registry.get(SfComponentType.STRATEGY_EXIT_RULE, "tp_sl")
            except KeyError:
                from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
                tpsl_cls = TakeProfitStopLossExit

            rules.append(tpsl_cls(
                take_profit_pct=tp or 0.02,
                stop_loss_pct=sl or 0.01,
            ))

        # Trailing stop
        trailing = self._exit_config.get("trailing")
        if trailing:
            try:
                trail_cls = default_registry.get(SfComponentType.STRATEGY_EXIT_RULE, "trailing")
                rules.append(trail_cls(trail_pct=trailing))
            except KeyError:
                pass

        # Time-based exit
        time_limit = self._exit_config.get("time_limit")
        if time_limit:
            try:
                time_cls = default_registry.get(SfComponentType.STRATEGY_EXIT_RULE, "time_based")
                rules.append(time_cls(max_bars=time_limit))
            except KeyError:
                pass

        # Default if nothing configured
        if not rules:
            from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
            rules.append(TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.01))

        return rules

    def _build_broker(self) -> Any:
        """Build broker from registry."""
        try:
            broker_cls = default_registry.get(SfComponentType.STRATEGY_BROKER, "backtest")
            executor_cls = default_registry.get(SfComponentType.STRATEGY_EXECUTOR, "virtual_spot")

            executor = executor_cls(fee_rate=self._fee)
            return broker_cls(executor=executor)
        except KeyError:
            # Fallback to direct imports
            from signalflow.strategy.broker import BacktestBroker
            from signalflow.strategy.broker.executor import VirtualSpotExecutor

            return BacktestBroker(executor=VirtualSpotExecutor(fee_rate=self._fee))

    def _build_runner(self, broker: Any, entry_rules: list[Any], exit_rules: list[Any]) -> Any:
        """Build runner from registry."""
        try:
            runner_cls = default_registry.get(SfComponentType.STRATEGY_RUNNER, "backtest")
        except KeyError:
            from signalflow.strategy.runner import BacktestRunner
            runner_cls = BacktestRunner

        return runner_cls(
            strategy_id=self.strategy_id,
            broker=broker,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            initial_capital=self._capital,
            show_progress=self._show_progress,
        )

    def visualize(
        self,
        *,
        output: str | None = None,
        format: str = "html",
        show: bool = True,
    ) -> str:
        """
        Visualize the configured pipeline.

        Opens an interactive HTML visualization showing the data flow
        from data sources through features to detector and runner.

        Args:
            output: Output file path (optional)
            format: Output format ("html" or "mermaid")
            show: Open in browser (HTML only)

        Returns:
            Rendered output string

        Example:
            >>> sf.Backtest("test").data(...).detector(...).visualize()
        """
        from signalflow import viz

        return viz.pipeline(self, output=output, format=format, show=show)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"BacktestBuilder(strategy_id={self.strategy_id!r})"


def Backtest(strategy_id: str = "backtest") -> BacktestBuilder:
    """
    Create a new backtest builder.

    This is the recommended way to configure and run backtests.

    Args:
        strategy_id: Unique identifier for the strategy

    Returns:
        BacktestBuilder instance for fluent configuration

    Example:
        >>> result = (
        ...     sf.Backtest("my_strategy")
        ...     .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        ...     .detector("example/sma_cross")
        ...     .run()
        ... )
    """
    return BacktestBuilder(strategy_id=strategy_id)
