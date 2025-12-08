# Configuration

SignalFlow uses a YAML-based configuration system for easy customization.

## Configuration File

Create a `signalflow.yml` in your project root:

```yaml
# signalflow.yml

project:
  name: "my-trading-project"
  version: "1.0.0"

data:
  provider: "binance"
  cache_dir: ".cache/data"
  default_timeframe: "1h"

backtest:
  initial_capital: 10000
  commission: 0.001  # 0.1%
  slippage: 0.0005   # 0.05%
  
execution:
  mode: "paper"  # paper, live
  exchange: "binance"
  
logging:
  level: "INFO"
  file: "logs/signalflow.log"

mlflow:
  tracking_uri: "mlruns"
  experiment_name: "trading-experiments"
```

## Environment Variables

Sensitive data should be stored in environment variables:

```bash
export SIGNALFLOW_API_KEY="your-api-key"
export SIGNALFLOW_API_SECRET="your-api-secret"
```

Or use a `.env` file:

```ini
SIGNALFLOW_API_KEY=your-api-key
SIGNALFLOW_API_SECRET=your-api-secret
```

## Loading Configuration

```python
from signalflow import Config

# Load from file
config = Config.from_file("signalflow.yml")

# Or load from dict
config = Config({
    "project": {"name": "quick-test"},
    "backtest": {"initial_capital": 5000}
})
```

## Configuration Options

### Data Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | `"yahoo"` | Data provider |
| `cache_dir` | string | `".cache"` | Cache directory |
| `default_timeframe` | string | `"1d"` | Default timeframe |

### Backtest Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `initial_capital` | float | `10000` | Starting capital |
| `commission` | float | `0.001` | Commission rate |
| `slippage` | float | `0.0` | Slippage rate |

### Execution Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | `"paper"` | Execution mode |
| `exchange` | string | `null` | Target exchange |

!!! warning "Live Trading"
    Always test thoroughly in paper mode before enabling live trading.

## GPU Configuration

Enable GPU acceleration:

```yaml
compute:
  device: "cuda"  # cpu, cuda, auto
  precision: "float32"  # float16, float32
```

```python
# Or configure programmatically
from signalflow import set_device

set_device("cuda")  # Use GPU
set_device("auto")  # Auto-detect
```