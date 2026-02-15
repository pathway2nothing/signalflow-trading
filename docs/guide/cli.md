# Command Line Interface

SignalFlow includes a CLI (`sf`) for running backtests from the command line.

---

## Installation

The CLI is automatically installed with the package:

```bash
pip install signalflow-trading
```

Verify installation:

```bash
sf --help
```

---

## Commands

### `sf init` - Create Config File

Generate a sample YAML configuration file:

```bash
# Create backtest.yaml (default)
sf init

# Custom filename
sf init --output my_strategy.yaml

# Overwrite existing file
sf init --force
```

### `sf validate` - Validate Config

Check configuration for errors before running:

```bash
sf validate backtest.yaml
```

Output:
```
Validating: backtest.yaml
Config is valid!
```

Or if there are issues:
```
Validating: backtest.yaml
WARNING: TP (1.0%) < SL (2.0%), risk/reward < 1
Config is valid (with warnings)
```

### `sf run` - Run Backtest

Execute a backtest from YAML config:

```bash
# Basic run
sf run backtest.yaml

# Save results to JSON
sf run backtest.yaml --output results.json

# Show plots after completion
sf run backtest.yaml --plot

# Quiet mode (no progress bar)
sf run backtest.yaml --quiet

# All options
sf run backtest.yaml -o results.json -p -q
```

Output:
```
Loading config: backtest.yaml
Running backtest: my_strategy

==================================================
         BACKTEST RESULT
==================================================
  Total Return:    +15.23%
--------------------------------------------------
  Trades:                  42
  Win Rate:             61.9%
  Profit Factor:         1.85
--------------------------------------------------
  Initial Capital:  $50,000.00
  Final Capital:    $57,615.00
  Max Drawdown:         -5.2%
  Sharpe Ratio:          1.42
==================================================
```

### `sf list` - List Components

Discover available components from the registry:

```bash
# List signal detectors
sf list detectors

# With detailed descriptions
sf list detectors --verbose

# List strategy metrics
sf list metrics

# List features
sf list features

# List all registered components
sf list all
```

Example output:
```
Available detectors (10):
----------------------------------------
  anomaly_detector
  example/sma_cross
  local_extrema_detector
  market_wide/agreement
  market_wide/cusum
  market_wide/zscore
  percentile_regime_detector
  structure_detector
  volatility_detector
  zscore_anomaly_detector
```

---

## YAML Configuration Reference

### Complete Example

```yaml
# backtest.yaml - Full configuration example

# Strategy identification
strategy:
  id: my_momentum_strategy

# Data source configuration
data:
  source: data/binance.duckdb   # Path to DuckDB file
  pairs:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
  start: "2024-01-01"           # ISO date string
  end: "2024-06-01"             # Optional, defaults to now
  timeframe: 1h                 # 1m, 5m, 15m, 1h, 4h, 1d
  data_type: perpetual          # spot | futures | perpetual

# Signal detector configuration
detector:
  name: example/sma_cross       # Registry name
  params:
    fast_period: 20
    slow_period: 50
    # Add any detector-specific parameters

# Entry rules
entry:
  size: 1000                    # Fixed size in quote currency
  size_pct: 0.1                 # OR percentage of capital (overrides size)
  max_positions: 5              # Maximum concurrent positions
  max_per_pair: 1               # Maximum positions per trading pair

# Exit rules
exit:
  tp: 0.03                      # Take profit: 3%
  sl: 0.015                     # Stop loss: 1.5%
  trailing: 0.02                # Trailing stop: 2% (optional)
  time_limit: 100               # Max bars to hold (optional)

# Capital and fees
capital: 50000                  # Initial capital
fee: 0.001                      # Trading fee: 0.1%

# Runtime options
show_progress: true             # Show progress bar
```

### Minimal Example

```yaml
# Minimum required configuration
data:
  source: data/prices.duckdb
  pairs: [BTCUSDT]
  start: "2024-01-01"

detector:
  name: example/sma_cross
```

### Configuration Sections

#### `strategy`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | `"backtest"` | Strategy identifier |

#### `data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Path to DuckDB file |
| `pairs` | list | Yes | Trading pairs to load |
| `start` | string | Yes | Start date (ISO format) |
| `end` | string | No | End date (defaults to now) |
| `timeframe` | string | No | Candle timeframe (default: `"1h"`) |
| `data_type` | string | No | Data type (default: `"perpetual"`) |

#### `detector`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Registry name (e.g., `"example/sma_cross"`) |
| `params` | object | No | Detector-specific parameters |

#### `entry`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rule` | string | `"signal"` | Entry rule name from registry |
| `size` | float | `100.0` | Fixed position size |
| `size_pct` | float | - | Position size as % of capital |
| `max_positions` | int | `10` | Max concurrent positions |
| `max_per_pair` | int | `1` | Max positions per pair |

#### `exit`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rule` | string | - | Exit rule name from registry |
| `tp` | float | `0.02` | Take profit percentage |
| `sl` | float | `0.01` | Stop loss percentage |
| `trailing` | float | - | Trailing stop percentage |
| `time_limit` | int | - | Max bars to hold |

#### Root Level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `capital` | float | `10000.0` | Initial capital |
| `fee` | float | `0.001` | Trading fee rate |
| `show_progress` | bool | `true` | Show progress bar |

---

## Using Custom Detectors

To use custom detectors in YAML config, first register them:

```python
# my_detectors.py
from signalflow.core import sf_component, SfComponentType
from signalflow.detector import SignalDetector

@sf_component(SfComponentType.DETECTOR, "my_namespace/rsi_crossover")
class RsiCrossoverDetector(SignalDetector):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def detect(self, features, context=None):
        # Detection logic
        ...
```

Then import before running CLI:

```python
# run_backtest.py
import my_detectors  # Register custom detectors

# Now run CLI programmatically
from signalflow.cli.main import cli
cli()
```

Or ensure the module is imported in your environment before running `sf`.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SF_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `SF_DATA_DIR` | Default data directory |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid config, runtime error, etc.) |

---

## Tips

### Quick Iteration

```bash
# Edit config, validate, run cycle
vim backtest.yaml
sf validate backtest.yaml && sf run backtest.yaml
```

### Comparing Strategies

```bash
# Run multiple configs
sf run strategy_a.yaml -o results_a.json
sf run strategy_b.yaml -o results_b.json

# Compare results with jq
jq '.metrics.total_return' results_a.json results_b.json
```

### Debugging

```bash
# Verbose output
sf list detectors -v

# Check available components
sf list all
```
