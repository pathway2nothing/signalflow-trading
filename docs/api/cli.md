# CLI

Command-line interface for running backtests and managing configurations.

!!! info "Entry Point"
    `sf` -- Installed automatically with the package via `pip install signalflow-trading`.

## Commands

### `sf run`

Run a backtest from a YAML configuration file.

```bash
sf run config.yaml                # Run backtest
sf run config.yaml -o results.json # Save results to JSON
sf run config.yaml --plot --quiet  # Show plots, suppress output
```

### `sf init`

Create a sample YAML configuration file in the current directory.

```bash
sf init                           # Creates backtest.yaml
```

### `sf validate`

Validate a configuration file without running the backtest.

```bash
sf validate config.yaml           # Check for errors
```

### `sf list`

List available components registered in the SignalFlow registry.

```bash
sf list detectors                 # List all detectors
sf list metrics                   # List all metrics
sf list features                  # List all features
sf list all                       # List everything
```

## YAML Configuration

### Basic Example

```yaml
strategy:
  name: sma_crossover

data:
  source: data/binance.duckdb
  pairs: [BTCUSDT, ETHUSDT]
  start: "2024-01-01"
  end: "2024-06-01"
  timeframe: 1h

detector:
  name: example/sma_cross
  params:
    fast_period: 20
    slow_period: 50

entry:
  size_pct: 0.1

exit:
  tp: 0.03
  sl: 0.015

capital: 50000
```

### Multi-Component Example

```yaml
strategy:
  name: ensemble

data:
  spot_1m:
    source: data/binance.duckdb
    pairs: [BTCUSDT, ETHUSDT]
    start: "2024-01-01"
    timeframe: 1m

detectors:
  fast_sma:
    name: example/sma_cross
    params:
      fast_period: 5
      slow_period: 15
  slow_sma:
    name: example/sma_cross
    params:
      fast_period: 20
      slow_period: 50

aggregation:
  mode: weighted
  weights: [0.6, 0.4]

entries:
  aggressive:
    source_detector: fast_sma
    size_pct: 0.05
  conservative:
    source_detector: slow_sma
    size_pct: 0.15

exits:
  tight:
    tp: 0.02
    sl: 0.01
  trailing:
    trailing: 0.03

capital: 50000
```

## API Reference

::: signalflow.cli.config.BacktestConfig
    options:
      show_root_heading: true
      show_source: false
      members: true

::: signalflow.cli.config.DataConfig
    options:
      show_root_heading: true
      show_source: false

::: signalflow.cli.config.DetectorConfig
    options:
      show_root_heading: true
      show_source: false

::: signalflow.cli.config.EntryConfig
    options:
      show_root_heading: true
      show_source: false

::: signalflow.cli.config.ExitConfig
    options:
      show_root_heading: true
      show_source: false
