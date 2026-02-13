# API Ergonomics Improvement Plan

> **Status:** Draft
> **Created:** 2024-02-13
> **Target Version:** v0.5.0
> **Estimated Time:** 10-12 days

## Overview

This plan describes improvements to SignalFlow's API to reduce verbosity and improve developer experience. The goal is to transform a 43-line boilerplate setup into a 12-line fluent Builder pattern.

## Key Principles

1. **Registry-first** — Use `SignalFlowRegistry` for component discovery and creation
2. **Leverage existing analytics** — `BacktestResult` wraps existing `StrategyMainResult`, `StrategyPairResult` metrics
3. **No breaking changes** — Lazy imports must not break `autodiscover()`
4. **Builder as primary API** — Fluent builder pattern for complex configurations

## Problem Statement

### Current State (v0.4.3)

A minimal backtest requires:
- **17 imports** from 8+ modules
- **43 lines of code**
- **9 parameter specifications** (most required, not defaults)
- Multiple wrapper calls (`.view()`, `.to_polars()`, `.value`)

```python
# Current verbose API (43 lines)
from datetime import datetime
from pathlib import Path
from signalflow.data.source import VirtualDataProvider
from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.data import RawDataFactory
from signalflow.detector import ExampleSmaCrossDetector
from signalflow.strategy.runner import BacktestRunner
from signalflow.strategy.component.entry.signal import SignalEntryRule
from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
from signalflow.strategy.broker import BacktestBroker
from signalflow.strategy.broker.executor import VirtualSpotExecutor

store = DuckDbSpotStore(db_path=Path("data.duckdb"))
raw_data = RawDataFactory.from_duckdb_spot_store(
    spot_store_path=Path("data.duckdb"),
    pairs=["BTCUSDT"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)
view = raw_data.view()

detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)
signals = detector.run(view)

entry_rule = SignalEntryRule(
    base_position_size=100.0,
    use_probability_sizing=False,
    max_positions_per_pair=1,
    max_total_positions=10,
)
exit_rule = TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.01)
broker = BacktestBroker(executor=VirtualSpotExecutor(fee_rate=0.001))

runner = BacktestRunner(
    strategy_id="my_strategy",
    broker=broker,
    entry_rules=[entry_rule],
    exit_rules=[exit_rule],
    initial_capital=10_000.0,
)
state = runner.run(raw_data=raw_data, signals=signals)
print(f"Trades: {len(runner.trades)}")
```

### Target State (Builder Pattern - Primary API)

```python
import signalflow as sf

result = (
    sf.Backtest("sma_cross_strategy")
    .data(exchange="binance", pairs=["BTCUSDT", "ETHUSDT"], start="2024-01-01")
    .detector("example/sma_cross", fast_period=20, slow_period=50)  # via registry
    .entry(size_pct=0.1, max_positions=5)
    .exit(tp=0.03, sl=0.015, trailing=0.02)
    .capital(50_000)
    .run()
)

# Rich result with existing analytics
print(result.summary())           # Uses StrategyMainResult metrics
result.plot()                     # Uses StrategyMainResult.plot()
result.plot_pair("BTCUSDT")       # Uses StrategyPairResult.plot()
print(result.metrics)             # All computed metrics dict
```

**Alternative: Instance-based detector**
```python
from signalflow.detector import ExampleSmaCrossDetector

result = (
    sf.Backtest("my_strategy")
    .data(raw=my_raw_data)  # Pre-loaded data
    .detector(ExampleSmaCrossDetector(fast_period=20, slow_period=50))  # Instance
    .run()
)
```

## Pain Points Addressed

| Pain Point | Current | Solution |
|------------|---------|----------|
| Multi-step data loading | Store → Provider → Factory → View → Polars | `sf.load()` one-liner |
| Broker/executor boilerplate | 3-level nesting, 5+ imports | Built into `sf.backtest()` |
| Entry/exit configuration | 4-5 required parameters each | Sensible defaults |
| Signal wrapper overhead | `.value` access required | Handled internally |
| Strategy ID requirement | Must specify | Auto-generated if not provided |
| DateTime parsing | `datetime()` required | String dates accepted |

## Architecture

```
src/signalflow/
├── __init__.py              # UPDATE: add lazy imports for API
├── api/                     # NEW: High-level API module
│   ├── __init__.py
│   ├── shortcuts.py         # sf.backtest(), sf.load()
│   ├── builder.py           # sf.Backtest() builder class
│   └── defaults.py          # Sensible default configurations
├── core/
├── data/
├── detector/
├── ...
```

## Implementation Phases

### Phase 1: BacktestResult with Existing Analytics (2-3 days)

**File:** `src/signalflow/api/result.py`

`BacktestResult` wraps existing analytics instead of reimplementing:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
import plotly.graph_objects as go

from signalflow.core import StrategyState, RawData, Signals, default_registry, SfComponentType

if TYPE_CHECKING:
    from signalflow.analytic.strategy import StrategyMainResult, StrategyPairResult


@dataclass
class BacktestResult:
    """
    Container for backtest results with convenient access to existing analytics.

    Wraps StrategyMainResult and StrategyPairResult from signalflow.analytic.strategy
    for visualization and metrics computation.
    """

    state: StrategyState
    trades: list[Any]
    signals: Signals
    raw: RawData
    config: dict
    metrics_df: pl.DataFrame | None = None

    # Cached analytics (created via registry)
    _main_result: StrategyMainResult | None = field(default=None, repr=False)
    _pair_result: StrategyPairResult | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize analytics from registry."""
        # Use registry to get metric classes
        try:
            main_cls = default_registry.get(SfComponentType.STRATEGY_METRIC, "result_main")
            self._main_result = main_cls()
        except KeyError:
            pass

    # === Basic Properties ===

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def final_capital(self) -> float:
        return self.state.capital

    @property
    def initial_capital(self) -> float:
        return self.config.get("capital", 10_000.0)

    @property
    def total_return(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if getattr(t, "pnl", 0) > 0)
        return wins / len(self.trades)

    # === Metrics (from existing analytics) ===

    @property
    def metrics(self) -> dict[str, float]:
        """Compute all metrics using registered STRATEGY_METRIC components."""
        results = {}
        prices = self._get_last_prices()

        # Get all registered metrics
        for name in default_registry.list(SfComponentType.STRATEGY_METRIC):
            try:
                metric_cls = default_registry.get(SfComponentType.STRATEGY_METRIC, name)
                metric = metric_cls()
                computed = metric.compute(self.state, prices)
                if computed:
                    results.update(computed)
            except Exception:
                pass

        return results

    # === Visualization (delegates to existing) ===

    def plot(self) -> list[go.Figure] | None:
        """Plot strategy results using StrategyMainResult."""
        if self._main_result is None:
            return None

        results_dict = self._build_results_dict()
        return self._main_result.plot(
            results=results_dict,
            state=self.state,
            raw_data=self.raw,
        )

    def plot_pair(self, pair: str) -> list[go.Figure] | None:
        """Plot pair-level results using StrategyPairResult."""
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

    # === Summary ===

    def summary(self) -> str:
        """Return formatted summary using computed metrics."""
        m = self.metrics

        return f"""
╔══════════════════════════════════════════╗
║         Backtest Summary                 ║
╠══════════════════════════════════════════╣
║ Trades:        {self.n_trades:>8}                  ║
║ Win Rate:      {self.win_rate:>8.1%}                  ║
║ Total Return:  {self.total_return:>+8.1%}                  ║
║ Final Capital: ${self.final_capital:>12,.2f}          ║
╠══════════════════════════════════════════╣
║ Max Drawdown:  {m.get('max_drawdown', 0):>8.1%}                  ║
║ Sharpe Ratio:  {m.get('sharpe_ratio', 0):>8.2f}                  ║
╚══════════════════════════════════════════╝
"""

    # === Helpers ===

    def _build_results_dict(self) -> dict:
        """Build results dict for analytics."""
        return {
            "metrics_df": self.metrics_df,
            "initial_capital": self.initial_capital,
            "final_return": self.total_return,
            "max_drawdown": self.metrics.get("max_drawdown", 0),
        }

    def _get_last_prices(self) -> dict[str, float]:
        """Get last prices from raw data."""
        prices = {}
        try:
            spot = self.raw.spot.to_polars() if hasattr(self.raw, 'spot') else None
            if spot is not None:
                for pair in spot["pair"].unique().to_list():
                    last = spot.filter(pl.col("pair") == pair).tail(1)
                    if last.height > 0:
                        prices[pair] = float(last["close"][0])
        except Exception:
            pass
        return prices
```

**Deliverables:**
- [ ] `BacktestResult` wrapping existing analytics
- [ ] `.metrics` property using registered STRATEGY_METRIC
- [ ] `.plot()` delegating to `StrategyMainResult`
- [ ] `.plot_pair()` delegating to `StrategyPairResult`
- [ ] `.summary()` with formatted output
- [ ] Unit tests

---

### Phase 2: Builder Pattern with Registry (3-4 days)

**File:** `src/signalflow/api/builder.py`

Builder uses registry for component creation:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Self, Any

from signalflow.core import (
    default_registry,
    SfComponentType,
    RawData,
    Signals,
)
from signalflow.api.result import BacktestResult

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.feature.base import Feature


@dataclass
class BacktestBuilder:
    """
    Fluent builder for backtest configuration.

    Uses SignalFlowRegistry for component discovery and creation.

    Example:
        >>> result = (
        ...     sf.Backtest("my_strategy")
        ...     .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        ...     .detector("example/sma_cross", fast_period=20, slow_period=50)
        ...     .entry(size_pct=0.1)
        ...     .exit(tp=0.03, sl=0.015)
        ...     .run()
        ... )
    """

    strategy_id: str

    # Internal state
    _raw: RawData | None = field(default=None, repr=False)
    _detector: SignalDetector | None = field(default=None, repr=False)
    _signals: Signals | None = field(default=None, repr=False)
    _entry_config: dict = field(default_factory=dict, repr=False)
    _exit_config: dict = field(default_factory=dict, repr=False)
    _capital: float = 10_000.0
    _fee: float = 0.001
    _show_progress: bool = True
    _data_params: dict | None = field(default=None, repr=False)

    # === Data Configuration ===

    def data(
        self,
        raw: RawData | None = None,
        *,
        exchange: str | None = None,
        pairs: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        timeframe: str = "1h",
    ) -> Self:
        """Configure data source (pre-loaded or lazy parameters)."""
        if raw is not None:
            self._raw = raw
        else:
            self._data_params = {
                "exchange": exchange,
                "pairs": pairs,
                "start": start,
                "end": end,
                "timeframe": timeframe,
            }
        return self

    # === Detector Configuration ===

    def detector(
        self,
        detector: SignalDetector | str,
        **kwargs: Any,
    ) -> Self:
        """
        Set signal detector (instance or registry name).

        Args:
            detector: SignalDetector instance OR registry name (e.g., "example/sma_cross")
            **kwargs: Parameters for registry-based creation

        Examples:
            >>> .detector(MyDetector(param=1))                    # Instance
            >>> .detector("example/sma_cross", fast_period=20)    # Registry
        """
        if isinstance(detector, str):
            # Create from registry
            self._detector = default_registry.create(
                SfComponentType.DETECTOR,
                detector,
                **kwargs,
            )
        else:
            self._detector = detector
        return self

    def signals(self, signals: Signals) -> Self:
        """Use pre-computed signals (skip detection)."""
        self._signals = signals
        return self

    # === Entry Configuration ===

    def entry(
        self,
        *,
        rule: str | None = None,  # Registry name
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
            max_positions: Maximum total positions
            max_per_pair: Maximum positions per pair
            **kwargs: Additional params for custom rule
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

    # === Exit Configuration ===

    def exit(
        self,
        *,
        rule: str | None = None,  # Registry name
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
            time_limit: Maximum bars to hold
            **kwargs: Additional params for custom rule
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

    # === Other Configuration ===

    def capital(self, amount: float) -> Self:
        """Set initial capital."""
        self._capital = amount
        return self

    def fee(self, rate: float) -> Self:
        """Set trading fee rate."""
        self._fee = rate
        return self

    def progress(self, show: bool = True) -> Self:
        """Enable/disable progress bar."""
        self._show_progress = show
        return self

    # === Execution ===

    def run(self) -> BacktestResult:
        """Execute backtest and return results."""
        # 1. Load data if needed
        raw = self._resolve_data()

        # 2. Get signals
        signals = self._resolve_signals(raw)

        # 3. Build components from registry
        entry_rules = self._build_entry_rules()
        exit_rules = self._build_exit_rules()
        broker = self._build_broker()

        # 4. Create and run runner (from registry)
        runner_cls = default_registry.get(SfComponentType.STRATEGY_RUNNER, "backtest")
        runner = runner_cls(
            strategy_id=self.strategy_id,
            broker=broker,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            initial_capital=self._capital,
            show_progress=self._show_progress,
        )

        state = runner.run(raw_data=raw, signals=signals)

        return BacktestResult(
            state=state,
            trades=runner.trades,
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

    # === Validation ===

    def validate(self) -> list[str]:
        """Validate configuration, return list of issues."""
        issues = []

        if self._raw is None and not self._data_params:
            issues.append("ERROR: No data source configured. Use .data()")

        if self._detector is None and self._signals is None:
            issues.append("ERROR: No detector or signals configured.")

        # Check registry availability
        try:
            default_registry.get(SfComponentType.STRATEGY_RUNNER, "backtest")
        except KeyError:
            issues.append("ERROR: BacktestRunner not found in registry")

        # Validate TP/SL ratio
        tp = self._exit_config.get("tp")
        sl = self._exit_config.get("sl")
        if tp and sl and tp < sl:
            issues.append(f"WARNING: TP ({tp}) < SL ({sl}), risk/reward < 1")

        return issues

    # === Private Helpers ===

    def _resolve_data(self) -> RawData:
        """Load data from params or return pre-loaded."""
        if self._raw is not None:
            return self._raw

        if not self._data_params:
            raise ValueError("No data configured. Use .data()")

        from signalflow.api.shortcuts import load
        return load(**{k: v for k, v in self._data_params.items() if v is not None})

    def _resolve_signals(self, raw: RawData) -> Signals:
        """Detect signals or return pre-computed."""
        if self._signals is not None:
            return self._signals

        if self._detector is None:
            raise ValueError("No detector configured.")

        return self._detector.run(raw.view())

    def _build_entry_rules(self) -> list:
        """Build entry rules from config using registry."""
        rule_name = self._entry_config.get("rule", "signal")

        try:
            rule_cls = default_registry.get(SfComponentType.STRATEGY_ENTRY_RULE, rule_name)
        except KeyError:
            # Fallback to SignalEntryRule
            from signalflow.strategy.component.entry.signal import SignalEntryRule
            rule_cls = SignalEntryRule

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

    def _build_exit_rules(self) -> list:
        """Build exit rules from config using registry."""
        rules = []

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

        return rules or self._build_default_exit_rules()

    def _build_default_exit_rules(self) -> list:
        """Build default exit rules if none configured."""
        from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
        return [TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.01)]

    def _build_broker(self):
        """Build broker from registry."""
        try:
            broker_cls = default_registry.get(SfComponentType.STRATEGY_BROKER, "backtest")
            executor_cls = default_registry.get(SfComponentType.STRATEGY_EXECUTOR, "virtual_spot")

            executor = executor_cls(fee_rate=self._fee)
            return broker_cls(executor=executor)
        except KeyError:
            # Fallback
            from signalflow.strategy.broker import BacktestBroker
            from signalflow.strategy.broker.executor import VirtualSpotExecutor

            return BacktestBroker(executor=VirtualSpotExecutor(fee_rate=self._fee))


def Backtest(strategy_id: str = "backtest") -> BacktestBuilder:
    """Create a new backtest builder."""
    return BacktestBuilder(strategy_id=strategy_id)
```

**Deliverables:**
- [ ] `BacktestBuilder` with registry-based component creation
- [ ] `.detector()` accepts string (registry name) or instance
- [ ] `.entry()` / `.exit()` use registry for rules
- [ ] `.validate()` checks registry availability
- [ ] `.run()` builds components from registry
- [ ] Unit tests

---

### Phase 3: Shortcuts Module (2 days)

**File:** `src/signalflow/api/shortcuts.py`

Lightweight shortcuts using registry:

```python
"""
Quick shortcuts for common operations.

These are thin wrappers around the Builder for simple use cases.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from signalflow.core import RawData, default_registry, SfComponentType

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.api.result import BacktestResult


def load(
    source: str | Path,
    *,
    pairs: list[str],
    start: str | datetime,
    end: str | datetime | None = None,
    timeframe: str = "1h",
    data_type: str = "perpetual",
) -> RawData:
    """
    Load market data from exchange or local file.

    Uses registry to find appropriate loader/store.

    Args:
        source: Exchange name or path to DuckDB file
        pairs: Trading pairs
        start: Start date (string or datetime)
        end: End date (default: now)
        timeframe: Candle timeframe
        data_type: Data type (spot, futures, perpetual)

    Examples:
        >>> raw = sf.load("binance", pairs=["BTCUSDT"], start="2024-01-01")
        >>> raw = sf.load("data.duckdb", pairs=["BTCUSDT"], start="2024-01-01")
    """
    from signalflow.data import RawDataFactory

    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end) if end else datetime.now()

    # Path = load from file
    if isinstance(source, Path) or (isinstance(source, str) and source.endswith(".duckdb")):
        return RawDataFactory.from_store(
            store_path=Path(source),
            pairs=pairs,
            start=start_dt,
            end=end_dt,
        )

    # String = load from exchange via registry
    return RawDataFactory.from_exchange(
        exchange=source,
        pairs=pairs,
        start=start_dt,
        end=end_dt,
        timeframe=timeframe,
        data_type=data_type,
    )


def backtest(
    detector: SignalDetector | str,
    *,
    raw: RawData | None = None,
    pairs: list[str] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    tp: float = 0.02,
    sl: float = 0.01,
    capital: float = 10_000.0,
    **kwargs,
) -> BacktestResult:
    """
    Quick backtest with minimal configuration.

    For more control, use sf.Backtest() builder.

    Examples:
        >>> result = sf.backtest(
        ...     detector="example/sma_cross",  # registry name
        ...     pairs=["BTCUSDT"],
        ...     start="2024-01-01",
        ... )
    """
    from signalflow.api.builder import Backtest

    builder = Backtest()

    # Data
    if raw is not None:
        builder.data(raw=raw)
    elif pairs and start:
        builder.data(pairs=pairs, start=start, end=end)

    # Detector (string = registry, instance = direct)
    if isinstance(detector, str):
        builder.detector(detector, **kwargs)
    else:
        builder.detector(detector)

    # Exit
    builder.exit(tp=tp, sl=sl)
    builder.capital(capital)

    return builder.run()


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)
```

**Deliverables:**
- [ ] `sf.load()` using registry for loaders
- [ ] `sf.backtest()` as thin wrapper around Builder
- [ ] Unit tests

---

### Phase 4: Update Main Init (1 day)

**File:** `src/signalflow/__init__.py`

**IMPORTANT:** Lazy imports must NOT break `autodiscover()`. The registry uses
`pkgutil.walk_packages()` which requires modules to be importable.

Strategy: Only lazy-load the `api` module, not core components:

```python
# At module level - these are always loaded (required for registry)
from signalflow.core import (
    RawData,
    Signals,
    # ... all existing imports ...
    default_registry,  # Ensure registry is available
)

# Existing module imports (required for autodiscover)
import signalflow.detector as detector
import signalflow.feature as feature
import signalflow.target as target
import signalflow.strategy as strategy
import signalflow.validator as validator
# ... etc ...


# ONLY lazy-load the new api module
def __getattr__(name: str):
    """
    Lazy load ONLY the api module.

    Note: We cannot lazy-load detector, feature, etc. because
    autodiscover() needs to walk those packages.
    """
    # New API shortcuts
    if name == "Backtest":
        from signalflow.api.builder import Backtest
        return Backtest

    if name == "BacktestResult":
        from signalflow.api.result import BacktestResult
        return BacktestResult

    if name == "backtest":
        from signalflow.api.shortcuts import backtest
        return backtest

    if name == "load":
        from signalflow.api.shortcuts import load
        return load

    raise AttributeError(f"module 'signalflow' has no attribute {name!r}")


# Update __all__
__all__ = [
    # ... existing exports ...

    # New API (lazy loaded)
    "Backtest",
    "BacktestResult",
    "backtest",
    "load",
]
```

**Deliverables:**
- [ ] Lazy imports ONLY for `api` module
- [ ] Verify `autodiscover()` still works
- [ ] Test: `default_registry.snapshot()` returns all components
- [ ] Updated `__all__`

---

### Phase 5: Tests (2-3 days)

**File:** `tests/api/test_result.py`

```python
"""Tests for BacktestResult with existing analytics."""
import pytest

from signalflow.api.result import BacktestResult
from signalflow.core import default_registry, SfComponentType


class TestBacktestResult:
    """Tests for BacktestResult wrapper."""

    def test_uses_registry_for_metrics(self, sample_state, sample_raw):
        """Metrics are computed via registered STRATEGY_METRIC components."""
        result = BacktestResult(
            state=sample_state,
            trades=[],
            signals=None,
            raw=sample_raw,
            config={"capital": 10_000},
        )

        metrics = result.metrics
        assert isinstance(metrics, dict)

    def test_plot_delegates_to_main_result(self, sample_result):
        """plot() uses StrategyMainResult from registry."""
        # Verify registry has the metric
        assert "result_main" in default_registry.list(SfComponentType.STRATEGY_METRIC)

        figs = sample_result.plot()
        # Should return figures or None (not crash)
        assert figs is None or isinstance(figs, list)

    def test_plot_pair_uses_registry(self, sample_result):
        """plot_pair() uses StrategyPairResult from registry."""
        assert "result_pair" in default_registry.list(SfComponentType.STRATEGY_METRIC)

        figs = sample_result.plot_pair("BTCUSDT")
        assert figs is None or isinstance(figs, list)

    def test_summary_format(self, sample_result):
        """summary() returns formatted string."""
        summary = sample_result.summary()
        assert "Trades:" in summary
        assert "Win Rate:" in summary
        assert "Total Return:" in summary


class TestBacktestResultProperties:
    """Tests for computed properties."""

    def test_n_trades(self, sample_result_with_trades):
        assert sample_result_with_trades.n_trades == 5

    def test_win_rate_calculation(self, sample_result_with_trades):
        # 3 wins out of 5
        assert sample_result_with_trades.win_rate == 0.6

    def test_total_return_calculation(self, sample_result):
        # (12000 - 10000) / 10000 = 0.2
        sample_result.state.capital = 12000
        assert sample_result.total_return == pytest.approx(0.2)
```

**File:** `tests/api/test_builder.py`

```python
"""Tests for BacktestBuilder with registry."""
import pytest

import signalflow as sf
from signalflow.core import default_registry, SfComponentType
from signalflow.detector import ExampleSmaCrossDetector


class TestBuilderRegistry:
    """Tests for registry-based component creation."""

    def test_detector_from_registry_name(self, sample_raw):
        """detector() accepts registry name string."""
        builder = (
            sf.Backtest("test")
            .data(raw=sample_raw)
            .detector("example/sma_cross", fast_period=10, slow_period=20)
        )

        assert builder._detector is not None
        assert builder._detector.fast_period == 10

    def test_detector_from_instance(self, sample_raw):
        """detector() accepts detector instance."""
        detector = ExampleSmaCrossDetector(fast_period=5, slow_period=15)

        builder = (
            sf.Backtest("test")
            .data(raw=sample_raw)
            .detector(detector)
        )

        assert builder._detector is detector

    def test_entry_uses_registry(self):
        """entry() looks up rules in registry."""
        # Verify registry has entry rules
        entry_rules = default_registry.list(SfComponentType.STRATEGY_ENTRY_RULE)
        assert len(entry_rules) > 0

    def test_exit_uses_registry(self):
        """exit() looks up rules in registry."""
        exit_rules = default_registry.list(SfComponentType.STRATEGY_EXIT_RULE)
        assert len(exit_rules) > 0

    def test_run_uses_registry_for_runner(self, sample_raw):
        """run() creates runner from registry."""
        # Verify BacktestRunner is registered
        assert "backtest" in default_registry.list(SfComponentType.STRATEGY_RUNNER)


class TestBuilderValidation:
    """Tests for configuration validation."""

    def test_validate_no_data(self):
        """Validation catches missing data."""
        builder = sf.Backtest("test")
        issues = builder.validate()
        assert any("No data" in i for i in issues)

    def test_validate_no_detector(self, sample_raw):
        """Validation catches missing detector."""
        builder = sf.Backtest("test").data(raw=sample_raw)
        issues = builder.validate()
        assert any("No detector" in i for i in issues)

    def test_validate_tp_sl_warning(self, sample_raw):
        """Validation warns when TP < SL."""
        builder = (
            sf.Backtest("test")
            .data(raw=sample_raw)
            .detector("example/sma_cross")
            .exit(tp=0.01, sl=0.02)  # Bad ratio
        )
        issues = builder.validate()
        assert any("WARNING" in i and "TP" in i for i in issues)


class TestBuilderFluent:
    """Tests for fluent API."""

    def test_methods_return_self(self, sample_raw):
        """All builder methods return self for chaining."""
        builder = sf.Backtest("test")

        assert builder.data(raw=sample_raw) is builder
        assert builder.detector("example/sma_cross") is builder
        assert builder.entry(size=100) is builder
        assert builder.exit(tp=0.02) is builder
        assert builder.capital(50_000) is builder
        assert builder.fee(0.002) is builder
        assert builder.progress(False) is builder


class TestBuilderRun:
    """Tests for backtest execution."""

    def test_run_returns_backtest_result(self, sample_raw):
        """run() returns BacktestResult instance."""
        result = (
            sf.Backtest("test")
            .data(raw=sample_raw)
            .detector("example/sma_cross")
            .run()
        )

        from signalflow.api.result import BacktestResult
        assert isinstance(result, BacktestResult)

    def test_run_with_full_config(self, sample_raw):
        """run() works with all options configured."""
        result = (
            sf.Backtest("full_test")
            .data(raw=sample_raw)
            .detector("example/sma_cross", fast_period=10, slow_period=30)
            .entry(size_pct=0.1, max_positions=5)
            .exit(tp=0.03, sl=0.015)
            .capital(50_000)
            .fee(0.0005)
            .progress(False)
            .run()
        )

        assert result.config["capital"] == 50_000
        assert result.config["fee"] == 0.0005
```

**File:** `tests/api/test_autodiscover.py`

```python
"""Tests to ensure lazy imports don't break autodiscover."""
import pytest


class TestAutodiscover:
    """Verify registry autodiscover still works with lazy imports."""

    def test_autodiscover_finds_detectors(self):
        """Autodiscover finds all @sf_component decorated detectors."""
        from signalflow.core import default_registry, SfComponentType

        detectors = default_registry.list(SfComponentType.DETECTOR)
        assert "example/sma_cross" in detectors

    def test_autodiscover_finds_metrics(self):
        """Autodiscover finds strategy metrics."""
        from signalflow.core import default_registry, SfComponentType

        metrics = default_registry.list(SfComponentType.STRATEGY_METRIC)
        assert "result_main" in metrics
        assert "result_pair" in metrics

    def test_autodiscover_finds_entry_rules(self):
        """Autodiscover finds entry rules."""
        from signalflow.core import default_registry, SfComponentType

        rules = default_registry.list(SfComponentType.STRATEGY_ENTRY_RULE)
        assert len(rules) > 0

    def test_autodiscover_finds_exit_rules(self):
        """Autodiscover finds exit rules."""
        from signalflow.core import default_registry, SfComponentType

        rules = default_registry.list(SfComponentType.STRATEGY_EXIT_RULE)
        assert len(rules) > 0

    def test_import_signalflow_triggers_discovery(self):
        """Importing signalflow makes registry available."""
        import signalflow as sf

        # Registry should be accessible
        assert hasattr(sf, 'default_registry')

        # And populated
        snapshot = sf.default_registry.snapshot()
        assert len(snapshot) > 0

    def test_lazy_api_import_doesnt_break_registry(self):
        """Accessing sf.Backtest doesn't break autodiscover."""
        import signalflow as sf

        # Access lazy-loaded API
        Backtest = sf.Backtest

        # Registry should still work
        from signalflow.core import SfComponentType
        detectors = sf.default_registry.list(SfComponentType.DETECTOR)
        assert len(detectors) > 0
```

**Deliverables:**
- [ ] Tests for `BacktestResult` with registry
- [ ] Tests for `BacktestBuilder` with registry
- [ ] Tests for `sf.load()` and `sf.backtest()`
- [ ] **Critical:** Tests that `autodiscover()` still works
- [ ] Fixtures in `conftest.py`

---

## File Structure

```
src/signalflow/
├── __init__.py              # UPDATE: lazy imports for api only
├── api/                     # NEW: High-level API module
│   ├── __init__.py          # Exports: Backtest, BacktestResult, load, backtest
│   ├── result.py            # BacktestResult wrapping existing analytics
│   ├── builder.py           # BacktestBuilder with registry integration
│   └── shortcuts.py         # sf.load(), sf.backtest() thin wrappers

tests/
├── api/                     # NEW: API tests
│   ├── __init__.py
│   ├── conftest.py          # Fixtures for API tests
│   ├── test_result.py       # BacktestResult tests
│   ├── test_builder.py      # BacktestBuilder tests
│   ├── test_shortcuts.py    # sf.load(), sf.backtest() tests
│   └── test_autodiscover.py # CRITICAL: Verify registry still works
```

## Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | `result.py` - BacktestResult with analytics | 2-3 days | - |
| 2 | `builder.py` - Builder with registry | 3-4 days | Phase 1 |
| 3 | `shortcuts.py` - Load/backtest wrappers | 2 days | Phase 2 |
| 4 | Update `__init__.py` - Lazy imports | 1 day | Phase 1-3 |
| 5 | Tests (including autodiscover) | 2-3 days | Phase 1-4 |
| **Total** | | **10-13 days** | |

## Registry Integration Points

Builder uses registry for these component types:

| Component | Registry Type | Registry Name | Fallback |
|-----------|---------------|---------------|----------|
| Detector | `DETECTOR` | e.g., `"example/sma_cross"` | Instance passed directly |
| Entry Rule | `STRATEGY_ENTRY_RULE` | e.g., `"signal"` | `SignalEntryRule` |
| Exit Rule | `STRATEGY_EXIT_RULE` | e.g., `"tp_sl"`, `"trailing"` | `TakeProfitStopLossExit` |
| Runner | `STRATEGY_RUNNER` | `"backtest"` | `BacktestRunner` |
| Broker | `STRATEGY_BROKER` | `"backtest"` | `BacktestBroker` |
| Executor | `STRATEGY_EXECUTOR` | `"virtual_spot"` | `VirtualSpotExecutor` |
| Metrics | `STRATEGY_METRIC` | `"result_main"`, `"result_pair"` | None |

## Autodiscover Safety

**Why lazy imports are limited:**

```
Registry.autodiscover()
    └── pkgutil.walk_packages("signalflow")
        └── imports all signalflow.* modules
            └── @sf_component decorators register components
```

If we lazy-load `signalflow.detector`, then `autodiscover()` won't find detectors.

**Solution:** Only lazy-load `signalflow.api` which is NEW and doesn't contain
any `@sf_component` decorated classes.

```python
# OK to lazy-load (new module, no components)
if name == "Backtest":
    from signalflow.api.builder import Backtest

# NOT OK to lazy-load (has @sf_component classes)
# import signalflow.detector  # Must stay eager
```

## Success Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines for minimal backtest | 43 | 12 | **72% reduction** |
| Required imports | 17 | 1 | **94% reduction** |
| Required parameters | 9 | 0 (all optional) | **100% reduction** |
| Time to first backtest | ~10 min | ~2 min | **80% reduction** |
| Registry utilization | Low | High | Components via registry |

## Backward Compatibility

All existing APIs remain unchanged. The new high-level API is purely additive:

- `signalflow.core.*` - unchanged
- `signalflow.data.*` - unchanged
- `signalflow.detector.*` - unchanged
- `signalflow.strategy.*` - unchanged
- `signalflow.analytic.*` - unchanged (used internally by BacktestResult)
- `default_registry` - unchanged (used by Builder)
- **NEW:** `signalflow.api.*` - new module
- **NEW:** `signalflow.Backtest()` - builder (lazy import)
- **NEW:** `signalflow.BacktestResult` - result wrapper (lazy import)
- **NEW:** `signalflow.load()` - data loader shortcut (lazy import)
- **NEW:** `signalflow.backtest()` - quick backtest (lazy import)

## Verification Checklist

Before merging, verify:

- [ ] `import signalflow` works without errors
- [ ] `sf.default_registry.snapshot()` shows all components
- [ ] `sf.Backtest()` is accessible (lazy loaded)
- [ ] Existing tests pass
- [ ] New API tests pass
- [ ] `autodiscover()` finds all `@sf_component` classes

## Future Extensions

After this plan is complete, consider:

1. **YAML Config Support** - Load strategy from YAML file
   ```python
   result = sf.backtest(config="strategy.yaml")
   ```

2. **CLI Integration** - Command-line interface
   ```bash
   sf backtest --config strategy.yaml
   ```

3. **Jupyter Widgets** - Interactive parameter tuning
   ```python
   sf.interactive_backtest(detector, raw)
   ```

## References

- [VectorBT API](https://vectorbt.dev/) - Inspiration for builder pattern
- [Backtrader Quickstart](https://www.backtrader.com/docu/quickstart/) - Comparison baseline
- [Freqtrade Config](https://www.freqtrade.io/en/stable/configuration/) - YAML config example
