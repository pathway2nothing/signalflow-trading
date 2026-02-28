---
title: Home
description: >
  SignalFlow is a high-performance Python framework for algorithmic trading,
  signal detection, meta-labeling validation, and strategy execution.

hide:
  - navigation
---

# Signalflow-trading - High-Performance Algorithmic Trading Framework

> Current stable version: **{{ project_version }}**

**signalflow-trading** is a high-performance Python framework for
**algorithmic trading**, **quantitative finance**, and
**ML-powered signal validation**.

---

## What's New in v0.6

<div class="grid cards" markdown>

-   :material-code-braces:{ .lg .middle } **Fluent Builder API**

    ---

    Clean, chainable API for backtesting with IDE autocomplete

    ```python
    result = (
        sf.Backtest("my_strategy")
        .data(raw=my_data)
        .detector("sma_cross")
        .exit(tp=0.03, sl=0.015)
        .run()
    )
    ```

-   :material-graph:{ .lg .middle } **Flow Builder**

    ---

    Execute strategies as directed flows with metric nodes, LTTB downsampling, and artifact caching

    ```python
    flow = (
        FlowBuilder("research")
        .data(store="binance", pair="BTC/USDT")
        .detector("sma_cross")
        .metric("sharpe_ratio")
        .build()
    )
    ```

-   :material-tag-multiple:{ .lg .middle } **Semantic Decorators**

    ---

    Type-safe component registration: `@sf.detector()`, `@sf.feature()`, `@sf.entry()`, `@sf.exit()`, `@sf.executor()`

-   :material-database:{ .lg .middle } **State Persistence**

    ---

    `StateManager` with Redis, DuckDB, and Memory backends for crash recovery and session continuity

-   :material-chart-scatter-plot:{ .lg .middle } **Statistical Analysis**

    ---

    Numba-accelerated Monte Carlo, Bootstrap CI, and significance tests

-   :material-console:{ .lg .middle } **CLI & YAML Config**

    ---

    Run backtests from command line with YAML configuration and `search_space()` for tuning

</div>

---

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **High Performance**

    ---

    Polars-first processing for 100+ pairs, 500k+ candles. Numba JIT for statistics

-   :material-puzzle:{ .lg .middle } **Modular Design**

    ---

    Component registry with pluggable detectors, validators, features, and strategies

-   :material-chart-line:{ .lg .middle } **Production Ready**

    ---

    Same code for backtesting, paper trading, and live execution

-   :material-brain:{ .lg .middle } **ML-Powered**

    ---

    scikit-learn, XGBoost, LightGBM, PyTorch signal validation with meta-labeling

</div>

---

## The Signal Pipeline

```mermaid
flowchart LR
    A[Market Data] --> B[Signal Detection]
    B --> C[Signal Validation]
    C --> D[Strategy Execution]

    style A fill:#14b8a6,stroke:#0d9488,stroke-width:2px,color:#fff
    style B fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff
    style C fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff
    style D fill:#22c55e,stroke:#16a34a,stroke-width:2px,color:#fff
```

### 1. Data & Features

- **7 Exchange Sources**: Binance, Bybit, OKX, Deribit, Kraken, Hyperliquid, WhiteBIT
- DuckDB, SQLite, PostgreSQL storage backends
- 189+ technical indicators via [signalflow-ta](ecosystem/signalflow-ta.md)

### 2. Signal Detection

- Classical algorithms (SMA crossover, MACD, RSI, structure detection)
- Neural network predictions via [signalflow-nn](ecosystem/signalflow-nn.md)
- Custom detectors via `@sf.detector("name")`

### 3. Signal Validation (Meta-Labeling)

- Lopez de Prado's meta-labeling methodology
- scikit-learn, XGBoost, LightGBM classifiers
- Deep learning validators via PyTorch Lightning

### 4. Strategy Execution

- **Entry/exit rules**: signal-based, model-driven, composite
- **Position sizing**: Kelly Criterion, volatility targeting, risk parity, martingale
- **Entry filters**: regime, volatility, drawdown, correlation, time-of-day
- **Signal aggregation**: majority, weighted, unanimous, meta-labeling
- **Real-time**: Paper trading with `RealtimeRunner`, monitoring, alerts
- **State persistence**: Redis/DuckDB/Memory with crash recovery

---

## Quick Example

```python
import signalflow as sf

result = (
    sf.Backtest("momentum_strategy")
    .data(raw=my_raw_data)
    .detector("example/sma_cross", fast_period=20, slow_period=50)
    .entry(size_pct=0.1, max_positions=5)
    .exit(tp=0.03, sl=0.015, trailing=0.02)
    .capital(50_000)
    .fee(0.001)
    .run()
)

print(result.summary())
result.plot()
```

### Custom Components

```python
import signalflow as sf
from signalflow.detector import SignalDetector

@sf.detector("my/custom_detector")
class CustomDetector(SignalDetector):
    def detect(self, data):
        return signals
```

Available decorators: `@sf.detector()`, `@sf.feature()`, `@sf.entry()`, `@sf.exit()`, `@sf.executor()`, `@sf.data_source()`, `@sf.data_store()`, `@sf.strategy_store()`, `@sf.register()`

---

## Key Features

### :octicons-zap-16: Polars-First Performance
Core data processing uses Polars for extreme efficiency on large datasets, with seamless Pandas compatibility.

### :octicons-beaker-16: Advanced Labeling
Triple Barrier Method, Fixed Horizon, Trend Scanning, Volatility and Volume labelers. Numba-accelerated (45s to 0.3s).

### :octicons-graph-16: 189+ Technical Indicators
The [signalflow-ta](ecosystem/signalflow-ta.md) extension: momentum, volatility, trend, statistics, and physics-based market analogs.

### :octicons-rocket-16: Paper Trading & Monitoring
`RealtimeRunner` with async data sync, virtual broker, drawdown/stuck-position alerts, and crash recovery.

### :octicons-cpu-16: Statistical Validation
Monte Carlo simulation, Bootstrap confidence intervals, significance tests — all Numba JIT-compiled.

### :octicons-plug-16: Flow Builder
Execute strategies as directed acyclic graphs with metric nodes, progress callbacks, and LTTB downsampling.

---

## Technology Stack

<div class="grid" markdown>

=== "Data Processing"
    - **Polars** — High-performance DataFrames
    - **DuckDB** — Embedded analytics database
    - **Pandas** — Compatibility layer
    - **NumPy** — Numerical computing

=== "Machine Learning"
    - **scikit-learn** — Classical ML models
    - **XGBoost / LightGBM** — Gradient boosting
    - **PyTorch + Lightning** — Deep learning
    - **Optuna** — Hyperparameter optimization
    - **Numba** — JIT compilation

=== "Trading"
    - **signalflow-ta** — 189+ indicators
    - **pandas-ta** — TA foundation
    - **Plotly** — Interactive charts

=== "Infrastructure"
    - **Redis** — State persistence (production)
    - **DuckDB / SQLite / PostgreSQL** — Storage
    - **Kedro** — Pipeline orchestration
    - **FastAPI** — Web backend (sf-ui)

</div>

---

## SignalFlow Ecosystem

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **signalflow-trading** (Core)

    ---

    Data containers, signal detection, backtesting, strategy execution, state persistence, statistical analysis

    ```bash
    pip install signalflow-trading
    ```

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[signalflow-ta](ecosystem/signalflow-ta.md)**

    ---

    189+ technical indicators, 24 signal detectors, physics-based market analogs, AutoFeatureNormalizer

    ```bash
    pip install signalflow-ta
    ```

-   :material-brain:{ .lg .middle } **[signalflow-nn](ecosystem/signalflow-nn.md)**

    ---

    14 neural encoders (LSTM, Transformer, PatchTST, TCN), 7 heads, 4 loss functions

    ```bash
    pip install signalflow-nn
    ```

-   :material-pipe:{ .lg .middle } **[sf-kedro](ecosystem/sf-kedro.md)**

    ---

    Universal ML pipelines: backtest, analyze, train, tune, validate with Optuna and MLflow


</div>

---

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **[Installation](getting-started/installation.md)**

    ---

    Install SignalFlow and set up your environment

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](quickstart.md)**

    ---

    Build your first strategy in 5 minutes

-   :material-tag-multiple:{ .lg .middle } **[Semantic Decorators](guide/semantic-decorators.md)**

    ---

    Register custom components with type-safe decorators

-   :material-strategy:{ .lg .middle } **[Advanced Strategies](guide/advanced-strategies.md)**

    ---

    Position sizing, entry filters, signal aggregation

-   :material-puzzle:{ .lg .middle } **[Ecosystem](ecosystem/index.md)**

    ---

    signalflow-ta, signalflow-nn, sf-kedro

-   :material-code-braces:{ .lg .middle } **[API Reference](api/index.md)**

    ---

    Complete documentation for all classes and methods

</div>

---

## Support

- **GitHub**: [github.com/pathway2nothing/signalflow-trading](https://github.com/pathway2nothing/signalflow-trading)
- **Issues**: [Report bugs or request features](https://github.com/pathway2nothing/signalflow-trading/issues)

---

## License

SignalFlow is open source software released under the [MIT License](https://opensource.org/licenses/MIT).

!!! warning "Disclaimer"
    SignalFlow is provided for research purposes. Trading financial instruments carries risk. Past performance does not guarantee future results. Use at your own risk.
