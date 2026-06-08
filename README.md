<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg" width="180">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo.png" width="180">
  <img alt="SignalFlow" src="docs/assets/logo.png" width="180">
</picture>

# SignalFlow

**Modular framework for trading signal detection, validation, and execution**

<p>
<a href="#quick-start">Quick Start</a>&ensp;&middot;&ensp;
<a href="#fluent-api">Fluent API</a>&ensp;&middot;&ensp;
<a href="#flow-builder">Flow Builder</a>&ensp;&middot;&ensp;
<a href="#cli">CLI</a>&ensp;&middot;&ensp;
<a href="https://signalflow-trading.com">Docs</a>
</p>

<p>
<a href="https://pypi.org/project/signalflow-trading/"><img src="https://img.shields.io/badge/version-0.6.0-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="License: MIT"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-1a1a2e" alt="Code style: ruff"></a>
<a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-3b82f6" alt="Type checked: mypy"></a>
</p>

<p>
<img src="https://img.shields.io/badge/polars-blueviolet" alt="Polars">
<img src="https://img.shields.io/badge/duckdb-1a1a2e" alt="DuckDB">
<img src="https://img.shields.io/badge/pytorch-ef4444?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/numba-00b4d8" alt="Numba">
</p>

</div>

---

SignalFlow is a high-performance Python framework for algorithmic trading that manages the full strategy lifecycle — from signal detection through meta-labeling validation to trade execution. It bridges the gap between research and production with a modular signal pipeline, fluent API, and visual DAG editor.

```
RawData → [Detector] → Signals → [Validator] → Validated → [Strategy] → Trades
```

## Quick Start

```bash
pip install signalflow-trading
pip install signalflow-ta    # 189+ technical indicators (optional)
pip install signalflow-nn    # neural network encoders (optional)
```

```python
import signalflow as sf

result = (
    sf.Backtest("my_strategy")
    .data(raw=my_raw_data)
    .detector("example/sma_cross", fast_period=20, slow_period=50)
    .entry(size_pct=0.1, max_positions=5)
    .exit(tp=0.03, sl=0.015)
    .capital(50_000)
    .run()
)

print(result.summary())
result.plot()
```

```
==================================================
           BACKTEST SUMMARY
==================================================
  Trades:                  42
  Win Rate:             61.9%
  Profit Factor:         1.85
--------------------------------------------------
  Initial Capital:  $50,000.00
  Final Capital:    $57,623.45
  Total Return:        +15.2%
--------------------------------------------------
  Max Drawdown:         -5.2%
  Sharpe Ratio:          1.42
==================================================
```

## Fluent API

Clean, chainable configuration with IDE autocomplete:

```python
import signalflow as sf

raw = sf.load(
    "data/binance.duckdb",
    pairs=["BTCUSDT", "ETHUSDT"],
    start="2024-01-01",
    end="2024-06-01",
)

result = (
    sf.Backtest("momentum_strategy")
    .data(raw=raw)
    .detector("example/sma_cross", fast_period=20, slow_period=50)
    .entry(size_pct=0.1, max_positions=5, max_per_pair=1)
    .exit(tp=0.03, sl=0.015, trailing=0.02)
    .capital(50_000)
    .fee(0.001)
    .run()
)

# Analyze
print(result.summary())
print(result.metrics)
result.plot()
result.plot_pair("BTCUSDT")

# Export
df = result.to_dataframe()
data = result.to_dict()
```

### Multi-Detector Ensembles

```python
result = (
    sf.Backtest("ensemble")
    .data(raw=spot_1m, name="1m")
    .data(raw=spot_1h, name="1h")
    .detector("sma_cross", name="trend", data_source="1h")
    .detector("volume_spike", name="volume", data_source="1m")
    .aggregation(mode="weighted", weights=[0.7, 0.3])
    .entry(size_pct=0.15)
    .exit(tp=0.03, sl=0.015)
    .capital(50_000)
    .run()
)
```

Aggregation modes: `majority`, `weighted`, `unanimous`, `any`, `meta_labeling`

## Flow Builder

Execute strategies as a DAG with multi-mode support, progress callbacks, and artifact caching:

```python
from signalflow.api import FlowBuilder

flow = (
    FlowBuilder("research_flow")
    .data(store="binance_futures", pair="BTC/USDT", timeframe="1h")
    .detector("sma_cross", fast=20, slow=50)
    .metric("total_return")
    .metric("sharpe_ratio")
    .metric("max_drawdown")
    .build()
)

result = flow.run()

# FlowResult provides equity curve, price data, detector features
result.equity_curve        # Polars DataFrame
result.price_data          # OHLCV with LTTB downsampling
result.detector_features   # Feature matrix from detector
result.metrics             # Computed metric values
```

## Semantic Decorators

Register custom components with type-safe decorators:

```python
import signalflow as sf

@sf.detector("my/custom_detector")
class MyDetector(BaseDetector):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def run(self, data: RawDataView) -> Signals:
        ...

@sf.feature("my/momentum")
class MomentumFeature(BaseFeature):
    ...

@sf.entry("my/aggressive_entry")
class AggressiveEntry(BaseEntryRule):
    ...

@sf.exit("my/trailing_exit")
class TrailingExit(BaseExitRule):
    ...
```

Available decorators: `@sf.detector()`, `@sf.feature()`, `@sf.entry()`, `@sf.exit()`, `@sf.executor()`, `@sf.data_source()`, `@sf.data_store()`, `@sf.strategy_store()`, `@sf.register()`

### Registry Discovery

```python
from signalflow.core import default_registry, SfComponentType

# List all available detectors
detectors = default_registry.list(SfComponentType.DETECTOR)

# Create from registry
detector = default_registry.create(
    SfComponentType.DETECTOR,
    "example/sma_cross",
    fast_period=20,
    slow_period=50,
)
```

## CLI

```bash
sf init                           # Create example YAML config
sf validate config.yaml           # Validate configuration
sf run config.yaml --plot         # Run backtest with plots
sf list detectors                 # List available components
sf viz config.yaml -o dag.html    # Visualize pipeline DAG
```

### YAML Configuration

```yaml
# backtest.yaml
strategy:
  id: my_strategy

data:
  source: data/binance.duckdb
  pairs: [BTCUSDT, ETHUSDT]
  start: "2024-01-01"
  end: "2024-06-01"
  timeframe: 1h
  data_type: perpetual

detector:
  name: example/sma_cross
  params:
    fast_period: 20
    slow_period: 50

entry:
  size_pct: 0.1
  max_positions: 5

exit:
  tp: 0.03
  sl: 0.015
  trailing: 0.02

capital: 50000
fee: 0.001
```

Components can define framework-agnostic hyperparameter search spaces via `search_space()` for Optuna / Kedro tuning.

## Signal Pipeline

### 1. Signal Detection

```python
from signalflow.detector import ExampleSmaCrossDetector

detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)
signals = detector.run(raw_data_view)
```

### 2. Signal Validation (Meta-Labeling)

Lopez de Prado methodology for signal quality assessment:

```python
from signalflow.validator import SklearnSignalValidator

validator = SklearnSignalValidator(model_type="random_forest")
validator.fit(X_train, y_train)
validated = validator.validate_signals(signals, features)
```

### 3. Strategy Execution

Multi-position model — each position is a discrete unit with one entry and one exit:

```
Timeline:
────────────────────────────────────────────────▶
  [P1: BUY──────SELL]
       [P2: BUY────────SELL]
            [P3: BUY───SELL]
                 [P4: BUY──────────SELL]
```

- Concurrent positions with overlap
- Natural support for averaging (multiple small positions)
- Atomic P&L attribution per trade

## State Persistence — event log is the source of truth

The portfolio changes ONLY through fills, so the system is event-sourced by
definition: the append-only **trade log** is the source of truth and the saved
state is a derived **snapshot cache**, verifiable by replay.

```python
from signalflow.core import fold
from signalflow.data.strategy_store import DuckDbStrategyStore

store = DuckDbStrategyStore("bot.duckdb")
store.init()

# Replay the event log into a portfolio (state = fold(events))
portfolio = fold(store.read_trades("my_bot"), initial_cash=10_000.0)

# Verify the snapshot cache equals the log replay
assert store.verify_snapshot("my_bot", initial_cash=10_000.0)
```

`StrategyStore` backends: **DuckDB** (single-node), **Memory** (backtesting),
plus optional Postgres. Durable live-trading state recovery (`StateManager`,
Redis backend) lives in the work-in-progress `signalflow.strategy.live` package
until live trading is wired.

## Statistical Analysis

Numba-accelerated statistics for strategy evaluation:

```python
from signalflow.analytic.stats import monte_carlo, bootstrap_ci

# Monte Carlo simulation of equity paths
mc_result = monte_carlo(trades, n_simulations=10_000)

# Bootstrap confidence intervals
ci = bootstrap_ci(returns, statistic="sharpe", confidence=0.95, n_bootstrap=5_000)
```

Includes: Monte Carlo simulation, Bootstrap CI, significance tests — all JIT-compiled with Numba.

## Strategy Components

| Category | Components |
|----------|-----------|
| **Position Sizing** | Kelly Criterion, Volatility Targeting, Risk Parity, Martingale, Signal Strength |
| **Entry Filters** | Regime, Volatility, Drawdown, Correlation, Time-of-Day |
| **Exit Rules** | TP/SL, Trailing Stop, Volatility Exit, Time-Based, Grid Exit, Composite |
| **Signal Aggregation** | Majority, Weighted, Unanimous, Any, Meta-Labeling |
| **Risk Management** | Position limits, Drawdown limits, Exposure limits |
| **Monitoring** | Drawdown alerts, Stuck position detection, Signal quality tracking |

## Package Structure

| Module | Description |
|--------|-------------|
| `signalflow.api` | Fluent Builder API (`Backtest`, `FlowBuilder`, `FlowResult`) |
| `signalflow.cli` | Command-line interface |
| `signalflow.config` | Flow configuration, YAML parsing, `ArtifactSchema` |
| `signalflow.core` | Data containers (`RawData`, `Signals`), registry, semantic decorators |
| `signalflow.data` | Exchange loaders, OHLCV resampling, DuckDB/SQLite/PostgreSQL stores |
| `signalflow.feature` | Feature extractors, `FeaturePipeline`, informativeness scoring |
| `signalflow.target` | Labeling (Triple Barrier, Fixed Horizon, Trend Scanning, Volatility, Volume) |
| `signalflow.detector` | Signal detection algorithms |
| `signalflow.validator` | ML-based signal validation (scikit-learn, LightGBM, XGBoost) |
| `signalflow.strategy` | Runners, brokers, entry/exit rules, sizing, state, monitoring, reconciliation |
| `signalflow.analytic` | Bootstrap, Monte Carlo, Numba-accelerated statistics |
| `signalflow.viz` | D3.js DAG visualization, Mermaid export |

## Supported Exchanges

| Exchange | Spot | Futures | Data Types |
|----------|------|---------|------------|
| Binance | ✅ | ✅ | OHLCV, Tick, Funding |
| Bybit | ✅ | ✅ (Linear & Inverse) | OHLCV, Tick |
| OKX | ✅ | ✅ | OHLCV |
| Deribit | — | ✅ | OHLCV, Options |
| Kraken | ✅ | ✅ | OHLCV |
| Hyperliquid | — | ✅ | OHLCV |
| WhiteBIT | ✅ | ✅ | OHLCV |

## Ecosystem

| Package | Description |
|---------|-------------|
| **[signalflow-ta](https://github.com/pathway2nothing/signalflow-ta)** | 189+ technical indicators, 24 signal detectors, physics-based analytics |
| **[signalflow-nn](https://github.com/pathway2nothing/signalflow-nn)** | 14 neural encoders (LSTM, Transformer, PatchTST), 7 classification heads |
| **[sf-kedro](https://github.com/pathway2nothing/sf-kedro)** | Kedro ML pipelines — backtest, tune, validate, train |
| **[sf-ui](https://github.com/pathway2nothing/sf-ui)** | Visual DAG editor — React 19, real-time backtesting |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Data** | Polars, DuckDB, pandas |
| **ML** | PyTorch, Lightning, scikit-learn, Numba, Optuna |
| **TA** | pandas-ta (via signalflow-ta) |
| **CLI** | Click, PyYAML |

---

**License:** MIT &ensp;·&ensp; **Author:** [pathway2nothing](https://github.com/pathway2nothing) &ensp;·&ensp; **Docs:** [signalflow-trading.com](https://signalflow-trading.com)
