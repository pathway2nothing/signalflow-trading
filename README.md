<div align="center">

<img src="docs/assets/logo.png" alt="SignalFlow" width="200">

# SignalFlow

### Modular framework for trading signal generation, validation and execution

<p>
<a href="#quick-start">Quick Start</a> •
<a href="#fluent-api">Fluent API</a> •
<a href="#cli">CLI</a> •
<a href="#core-architecture-the-signal-pipeline">Architecture</a> •
<a href="https://signalflow-trading.com">Docs</a>
</p>

<p>
<a href="https://github.com/pathway2nothing/signalflow-trading"><img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python 3.12+"></a>
<a href="https://github.com/pathway2nothing/signalflow-trading/releases"><img src="https://img.shields.io/badge/version-0.5.0-orange" alt="Version 0.5.0"></a>
<a href="https://github.com/pathway2nothing/signalflow-trading/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Code style: ruff"></a>
<a href="https://github.com/pathway2nothing/signalflow-trading"><img src="https://img.shields.io/badge/type%20checked-mypy-blue" alt="Type checked: mypy"></a>
</p>

<p>
<a href="https://github.com/pathway2nothing/signalflow-trading"><img src="https://img.shields.io/badge/data-polars%20%7C%20duckdb-blueviolet" alt="Data: Polars | DuckDB"></a>
<a href="https://github.com/pathway2nothing/signalflow-trading"><img src="https://img.shields.io/badge/ML-pytorch%20%7C%20lightning-red?logo=pytorch&logoColor=white" alt="ML: PyTorch | Lightning"></a>
<a href="https://github.com/pathway2nothing/signalflow-trading"><img src="https://img.shields.io/badge/exchanges-8%20supported-yellow" alt="Exchanges: 8 supported"></a>
</p>

</div>

---

**SignalFlow** is a high-performance Python framework for algorithmic trading, designed to manage the full strategy lifecycle from signal detection to execution. It bridges the gap between research and production by providing a robust pipeline for signal generation, meta-labeling validation, and automated trading.

## Quick Start

### Installation

```bash
pip install signalflow-trading
pip install signalflow-ta    # 199+ technical indicators (optional)
pip install signalflow-nn    # neural network validators (optional)
```

### Run Your First Backtest

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

**Output:**
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

SignalFlow v0.5 introduces a clean, chainable API for backtesting:

```python
import signalflow as sf

# Load data
raw = sf.load(
    "data/binance.duckdb",
    pairs=["BTCUSDT", "ETHUSDT"],
    start="2024-01-01",
    end="2024-06-01",
)

# Configure and run backtest
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

# Analyze results
print(result.summary())      # Text summary
print(result.metrics)        # All metrics as dict
result.plot()                # Interactive plots
result.plot_pair("BTCUSDT")  # Pair-level analysis

# Export
df = result.to_dataframe()   # Trades as Polars DataFrame
data = result.to_dict()      # Full results as dict
```

### Use Custom Detector

```python
from signalflow.detector import ExampleSmaCrossDetector

# Create detector instance
detector = ExampleSmaCrossDetector(fast_period=10, slow_period=30)

# Pass to builder
result = (
    sf.Backtest("custom_detector")
    .data(raw=raw)
    .detector(detector)  # Instance instead of registry name
    .exit(tp=0.02, sl=0.01)
    .run()
)
```

### Registry-Based Components

All components are discoverable via the registry:

```python
from signalflow.core import default_registry, SfComponentType

# List available detectors
detectors = default_registry.list(SfComponentType.DETECTOR)
print(detectors)
# ['example/sma_cross', 'volatility_detector', 'structure_detector', ...]

# Create component from registry
detector = default_registry.create(
    SfComponentType.DETECTOR,
    "example/sma_cross",
    fast_period=20,
    slow_period=50,
)
```

## CLI

SignalFlow includes a command-line interface for quick operations:

```bash
# Create sample config
sf init

# Validate configuration
sf validate backtest.yaml

# Run backtest from YAML
sf run backtest.yaml
sf run backtest.yaml --output results.json --plot

# List available components
sf list detectors
sf list metrics
sf list all
```

### YAML Configuration

```yaml
# backtest.yaml
strategy:
  id: my_strategy

data:
  source: data/binance.duckdb
  pairs:
    - BTCUSDT
    - ETHUSDT
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
  max_per_pair: 1

exit:
  tp: 0.03
  sl: 0.015
  trailing: 0.02

capital: 50000
fee: 0.001
```

## Multi-Detector Ensembles

Combine multiple detectors with signal aggregation:

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

**Aggregation modes:** `majority`, `weighted`, `unanimous`, `any`, `meta_labeling`

## Flow-Based Execution (DAG)

Execute strategies as a DAG with multi-mode support, progress callbacks, and artifact caching:

```python
from signalflow.config import Flow, FlowMode, ArtifactCache

# Define flow as DAG
flow = Flow.from_dict({
    "id": "my_flow",
    "nodes": {
        "loader": {"type": "data/loader", "config": {"pairs": ["BTCUSDT"]}},
        "detector": {"type": "signals/detector", "name": "example/sma_cross"},
        "strategy": {"type": "strategy", "config": {...}},
    }
})

# Execute with caching and progress tracking
cache = ArtifactCache(cache_mode="disk", cache_dir="./cache")

result = flow.run(
    mode=FlowMode.BACKTEST,
    progress_callback=lambda step, total, info: print(f"{step}/{total}: {info['node']}"),
    cache=cache,
)

# Access artifacts
print(result.artifacts.keys())  # ['loader.ohlcv', 'detector.signals', ...]
print(result.execution_time)    # 2.5 (seconds)
print(cache.stats)              # {'mode': 'disk', 'entries': 3}
```

**Execution modes:** `FlowMode.BACKTEST`, `FlowMode.TRAIN`, `FlowMode.ANALYZE`

### Artifact Schema Validation

Type-safe validation for data flowing between nodes:

```python
from signalflow.config import OHLCV_SCHEMA, SIGNALS_SCHEMA

# Validate DataFrame against schema
errors = OHLCV_SCHEMA.validate(ohlcv_df)
if errors:
    print("Validation errors:", errors)

# Runtime validation during flow execution
result = flow.run(mode=FlowMode.BACKTEST, validate_runtime=True)
```

## Core Architecture: The Signal Pipeline

The framework implements a modular three-stage processing logic:

1. **Signal Detector**: Scans market data (OHLCV or tick) to identify potential market events. Detectors can range from simple SMA crossovers to complex deep learning models.

2. **Signal Validator (Meta-Labeling)**: Based on Lopez de Prado's methodology, this stage assesses the quality and risk of detected signals using classification models (e.g., LightGBM, XGBoost).

3. **Trading Strategy**: Converts validated signals into actionable trade positions, managing entry, exit, and risk.

## Key Features

* **Fluent Builder API**: Clean, chainable configuration for backtests with IDE autocomplete support.

* **CLI & YAML Config**: Run backtests from command line with YAML configuration files.

* **Polars-First Performance**: Core data processing utilizes `polars` for extreme efficiency with large datasets.

* **Production Ready**: Code written for research and backtesting is designed for direct deployment to live trading.

* **Helpful Error Messages**: Clear, actionable error messages with suggestions for fixing issues.

* **Jupyter Support**: Rich HTML rendering of results in Jupyter notebooks.

* **Advanced Strategy Components**:
  - **Position Sizing**: Kelly Criterion, volatility targeting, risk parity, martingale/grid strategies
  - **Entry Filters**: Regime, volatility, drawdown, correlation, time-of-day filtering
  - **Exit Rules**: Trailing stops, volatility-based exits, composite exit managers
  - **Signal Aggregation**: Majority voting, weighted averaging, meta-labeling

* **OHLCV Resampling**: Unified timeframe conversion across 8 exchanges with auto-detection and smart timeframe selection.

* **Flow-Based Execution**: DAG-based strategy execution with multi-mode support (backtest, train, analyze), progress callbacks, cancellation, and artifact caching.

* **Paper Trading Ready**: Real-time `RealtimeRunner` with monitoring, alerts, and virtual execution for risk-free validation.

* **ML/RL Integration**: Protocol-based external model integration with `ModelEntryRule` and `ModelExitRule` for automated decision-making.

* **Advanced Labeling**: Native support for Triple-Barrier Method and Fixed-Horizon labeling for ML training.

* **Strategy Monitoring**: Built-in alert system (drawdown, stuck positions, signal quality) for real-time oversight.

* **Flexible Extensibility**: Easily add custom features via the `@sf_component` registry.


## Signal Detection Example

```python
from signalflow.detector import ExampleSmaCrossDetector

# Initialize a detector (SMA 20/50 crossover)
detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)

# Run detection on a data snapshot
signals = detector.run(raw_data_view)
```

## Signal Validation (Meta-Labeling)

```python
from signalflow.validator import SklearnSignalValidator

# Create a validator using Random Forest
validator = SklearnSignalValidator(model_type="random_forest")

# Fit the model on labeled historical signals
validator.fit(X_train, y_train)

# Validate new signals to get success probabilities
validated_signals = validator.validate_signals(signals, features)
```

## Tech Stack

* **Data**: `polars`, `pandas`, `duckdb`
* **ML/Compute**: `pytorch`, `lightning`, `scikit-learn`, `numba`, `optuna`
* **Technical Analysis**: `pandas-ta`
* **CLI**: `click`, `pyyaml`


## Package Structure

* `signalflow.api`: High-level fluent API (`Backtest`, `BacktestResult`, `load`)
* `signalflow.cli`: Command-line interface
* `signalflow.config`: Flow configuration (`Flow`, `FlowMode`, `ArtifactCache`, `ArtifactSchema`)
* `signalflow.core`: Core data containers (`RawData`, `Signals`) and registries
* `signalflow.data`: Exchange loaders (Binance, Bybit, OKX, Deribit, Kraken, Hyperliquid, WhiteBIT), OHLCV resampling, DuckDB/SQLite/PostgreSQL storage
* `signalflow.feature`: Feature extractors and technical indicator adapters
* `signalflow.target`: Advanced labeling techniques (Fixed Horizon, Triple Barrier)
* `signalflow.detector`: Signal detection algorithms
* `signalflow.validator`: ML-based signal validation (scikit-learn, LightGBM, XGBoost)
* `signalflow.strategy`:
  - **Runners**: `BacktestRunner`, `RealtimeRunner` for paper/live trading
  - **Components**: Position sizing, entry filters, exit rules, signal aggregation
  - **Model Integration**: Protocol-based ML/RL model integration
  - **Monitoring**: Real-time alerts and performance tracking
  - **Brokers**: Backtest, virtual, and live execution brokers

## Ecosystem

* **[signalflow-ta](https://github.com/pathway2nothing/signalflow-ta)**: 199+ technical analysis indicators (momentum, volatility, trend, statistics, physics-based)
* **[signalflow-nn](https://github.com/pathway2nothing/signalflow-nn)**: Neural network validators (LSTM, GRU, Attention) via PyTorch Lightning


---

**License:** MIT

**Author:** pathway2nothing

---
