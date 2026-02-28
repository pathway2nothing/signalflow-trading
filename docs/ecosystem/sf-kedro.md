---
title: sf-kedro
description: Universal ML pipelines for SignalFlow — Kedro-based backtesting, optimization, and validation
---

# sf-kedro — Universal ML Pipelines

**sf-kedro** implements a **Universal Pipelines Architecture** on top of
[Kedro](https://kedro.org/). Pipelines are defined by purpose — not by strategy
name — so the same `backtest`, `tune`, or `validate` pipeline works with any
flow configuration.

---

## Installation

```bash
conda create --name sf-kedro python==3.12
conda activate sf-kedro
pip install -r requirements.txt
cp .env.example .env
```

Requires `signalflow-trading`, `signalflow-ta`, `signalflow-nn`, `kedro >=1.1`.

---

## Architecture

```
FLOW CONFIG (conf/base/flows/*.yml)
├── detector   (required) → signal generation
├── validator  (optional) → ML signal filtering
└── strategy   (optional) → entry/exit rules

UNIVERSAL PIPELINES
├── backtest   → run backtest for any flow
├── analyze    → explore features and signals
├── train      → train validator model
├── tune       → Optuna parameter optimization
└── validate   → walk-forward validation
```

---

## Quick Start

```bash
# Run a backtest
kedro run --pipeline=backtest --params='flow_id=grid_sma'

# Analyze features & signals
kedro run --pipeline=analyze --params='flow_id=grid_sma'

# Optuna hyperparameter optimization
kedro run --pipeline=tune --params='flow_id=grid_sma,n_trials=100'

# Walk-forward validation
kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'

# Train ML validator
kedro run --pipeline=train --params='flow_id=grid_sma'
```

---

## Pipelines

### backtest

Run a backtest for any flow configuration.

**Nodes:** `load_flow_data` → `run_flow_detection` → `run_flow_backtest` → `compute_metrics` → `save_flow_plots`

```
==================================================
Backtest Complete: Grid SMA Crossover
--------------------------------------------------
  Initial Capital: $10,000.00
  Final Equity:    $9,662.57
  Total Return:    -3.37%
  Trades Executed: 756
  Win Rate:        34.6%
  Max Drawdown:    3.66%
==================================================
```

### analyze

Feature exploration and signal quality analysis.

```bash
kedro run --pipeline=analyze --params='flow_id=grid_sma,level=signals'
```

Levels: `features`, `signals`, `all`

### train

Train an ML validator for signal filtering.

**Nodes:** `load_training_data` → `prepare_features` → `train_validator` → `save_model`

### tune

Optuna hyperparameter optimization with configurable search spaces.

```bash
kedro run --pipeline=tune --params='flow_id=grid_sma,n_trials=100,level=strategy'
```

Levels: `detector`, `strategy`

### validate

Walk-forward out-of-sample validation.

```bash
kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'
```

```
==================================================
Walk-Forward Validation: Grid SMA Crossover
--------------------------------------------------
  Valid folds:     5/5
  Avg Return:      +1.23%
  Total trades:    1250
==================================================
```

---

## Flow Configuration

Flows are defined in YAML and passed to any pipeline via `flow_id`:

```yaml
# conf/base/flows/grid_sma.yml
flow_id: grid_sma
flow_name: "Grid SMA Crossover"

data:
  pairs: [BTCUSDT, ETHUSDT]

detector:
  type: "example/sma_cross"
  fast_period: 60
  slow_period: 720

strategy:
  entry_rules:
    - type: "signal"
      base_position_size: 200.0
      max_positions_per_pair: 5
      entry_filters:
        - type: "price_distance_filter"
          min_distance_pct: 0.02
  exit_rules:
    - type: "tp_sl"
      take_profit_pct: 0.015
      stop_loss_pct: 0.01
  metrics:
    - type: "total_return"
    - type: "win_rate"
    - type: "sharpe_ratio"
    - type: "drawdown"
    - type: "profit_factor"
```

---

## Project Structure

```
sf-kedro/
├── conf/base/
│   ├── parameters/          # Pipeline-specific params
│   │   ├── common.yml       # Shared defaults
│   │   ├── backtest.yml
│   │   ├── analyze.yml
│   │   ├── train.yml
│   │   ├── tune.yml
│   │   └── validate.yml
│   ├── flows/               # Flow configs
│   │   └── grid_sma.yml
│   └── catalog/             # Data catalog
├── src/sf_kedro/
│   ├── pipelines/
│   │   ├── backtest/
│   │   ├── analyze/
│   │   ├── train/
│   │   ├── tune/
│   │   └── validate/
│   └── utils/
│       ├── flow_config.py
│       ├── detection.py
│       └── telegram.py
└── data/
```

---

## Integrations

| Integration | Purpose |
|-------------|---------|
| **MLflow / DagsHub** | Experiment tracking, model registry |
| **Optuna** | Hyperparameter optimization |
| **Telegram** | Automated notifications |
| **Plotly** | Interactive visualizations |

---

## Links

- [:material-github: GitHub Repository](https://github.com/pathway2nothing/sf-kedro)
