# Configuration API

The `signalflow.config` module provides unified configuration loading and DAG-based flow definitions for all SignalFlow frontends (sf-kedro, sf-ui, CLI).

---

## Quick Start

```python
import signalflow as sf

# Load flow config from YAML
config = sf.config.load("grid_sma", conf_path="./conf")

# Run backtest from config
result = sf.Backtest.from_dict(config).run()

# Or use typed FlowConfig
flow = sf.config.FlowConfig.from_dict(config)
print(flow.detector.type)  # "example/sma_cross"
```

---

## Module Structure

```
signalflow.config
├── loader.py       # YAML loading utilities
├── flow.py         # FlowConfig dataclass (chain-style)
└── dag.py          # FlowDAG (DAG-style with auto-inference)
```

---

## Configuration Loading

### load_flow_config

```python
def load_flow_config(
    flow_id: str,
    conf_path: Path | str | None = None,
    *,
    resolve_env: bool = True
) -> dict[str, Any]
```

Load and merge flow configuration from YAML files.

**Parameters:**

- `flow_id` - Flow identifier (e.g., `"grid_sma"`)
- `conf_path` - Path to configuration directory (default: auto-detect)
- `resolve_env` - Whether to resolve `${ENV_VAR}` placeholders

**Returns:** Merged configuration dictionary

**Example:**

```python
config = sf.config.load("grid_sma")
# Merges: parameters/common.yml + flows/grid_sma.yml
```

### list_flows

```python
def list_flows(conf_path: Path | str | None = None) -> list[str]
```

List available flow configurations.

```python
flows = sf.config.list_flows("./conf")
# ["grid_sma", "baseline_sma", "rsi_momentum", ...]
```

### deep_merge

```python
def deep_merge(base: dict, override: dict) -> dict
```

Deep merge two dictionaries. Override values take precedence.

```python
base = {"a": 1, "nested": {"x": 10}}
override = {"nested": {"y": 20}}
result = sf.config.deep_merge(base, override)
# {"a": 1, "nested": {"x": 10, "y": 20}}
```

---

## FlowConfig (Chain-Style)

For simple flows with a linear structure: `data → detector → strategy`.

### FlowConfig

```python
@dataclass
class FlowConfig:
    flow_id: str
    flow_name: str = ""
    description: str = ""
    data: DataConfig
    detector: DetectorConfig | None
    strategy: StrategyConfig
    capital: float = 10000.0
    fee: float = 0.001
    raw: dict[str, Any]  # Original config
```

**Example:**

```python
from signalflow.config import FlowConfig

flow = FlowConfig.from_dict({
    "flow_id": "my_strategy",
    "detector": {
        "type": "example/sma_cross",
        "fast_period": 20,
        "slow_period": 50,
    },
    "strategy": {
        "entry_rules": [{"type": "signal", "base_position_size": 100}],
        "exit_rules": [{"type": "tp_sl", "take_profit_pct": 0.02}],
    },
})

# Access typed config
print(flow.detector.type)  # "example/sma_cross"
print(flow.strategy.entry_rules[0].base_position_size)  # 100.0

# Convert to BacktestBuilder format
config = flow.to_backtest_config()
result = sf.Backtest.from_dict(config).run()
```

### Nested Config Classes

```python
@dataclass
class DataConfig:
    pairs: list[str] = ["BTCUSDT"]
    timeframe: str = "1h"
    store: dict[str, Any]
    period: dict[str, Any]

@dataclass
class DetectorConfig:
    type: str  # Registry name (e.g., "example/sma_cross")
    params: dict[str, Any]

@dataclass
class StrategyConfig:
    strategy_id: str = "backtest"
    entry_rules: list[EntryRuleConfig]
    exit_rules: list[ExitRuleConfig]
    metrics: list[str]

@dataclass
class EntryRuleConfig:
    type: str = "signal"
    base_position_size: float = 100.0
    max_positions_per_pair: int = 1
    max_total_positions: int = 10
    entry_filters: list[EntryFilterConfig]
    params: dict[str, Any]

@dataclass
class ExitRuleConfig:
    type: str = "tp_sl"
    params: dict[str, Any]

@dataclass
class EntryFilterConfig:
    type: str  # Registry name (e.g., "price_distance_filter")
    params: dict[str, Any]
```

---

## FlowDAG (DAG-Style)

For complex flows with multiple data sources, parallel detectors, validators, and sophisticated strategy structures.

### Overview

```python
from signalflow.config import FlowDAG

dag = FlowDAG.from_dict({
    "nodes": {
        "loader": {"type": "data/loader", "config": {"exchange": "binance"}},
        "detector": {"type": "signals/detector", "name": "sma_cross"},
        "strategy": {"type": "strategy"},
    }
})

# Edges auto-inferred from inputs/outputs
dag.edges  # [Edge(loader → detector), Edge(detector → strategy)]
```

### Node

```python
@dataclass
class Node:
    id: str                          # Unique identifier
    type: str                        # Component type
    name: str = ""                   # Registry name
    config: dict[str, Any]           # Node-specific config
    inputs: list[str] | None = None  # Explicit inputs (auto if None)
    outputs: list[str] | None = None # Explicit outputs (auto if None)
    training_only: bool = False      # Only for validator training
    store: dict[str, Any] | None     # Store config for loaders
    tags: list[str]                  # Arbitrary tags
```

**Component Types:**

| Type | Default Inputs | Default Outputs |
|------|----------------|-----------------|
| `data/loader` | `[]` | `["ohlcv"]` |
| `feature` | `["ohlcv"]` | `["features"]` |
| `signals/detector` | `["ohlcv"]` | `["signals"]` |
| `signals/labeler` | `["ohlcv", "signals"]` | `["labels"]` |
| `signals/validator` | `["signals", "labels"]` | `["validated_signals"]` |
| `strategy` | `["ohlcv", "signals"]` | `["trades", "metrics"]` |

### Edge

```python
@dataclass
class Edge:
    source: str      # Source node ID
    target: str      # Target node ID
    data_type: str   # Data flowing (e.g., "ohlcv", "signals")
```

### FlowDAG

```python
@dataclass
class FlowDAG:
    id: str
    name: str = ""
    nodes: dict[str, Node]
    edges: list[Edge]
    config: dict[str, Any]  # Global config (capital, fee, etc.)
```

**Methods:**

```python
# Create from dict (auto-infers edges)
dag = FlowDAG.from_dict(config_dict)

# Get nodes by type
dag.get_loaders()                    # All data/loader nodes
dag.get_detectors()                  # All signals/detector nodes
dag.get_detectors(include_training_only=False)
dag.get_validators()                 # All signals/validator nodes
dag.get_strategy_node()              # The strategy node

# Execution
dag.topological_sort()               # Nodes in execution order
dag.get_execution_plan()             # Detailed execution plan

# Validation
errors = dag.validate()              # List of validation errors

# Serialization
dag.to_dict()                        # Convert to dict
```

### Auto-Edge Inference

Edges are automatically created based on node inputs/outputs:

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    dag = FlowDAG.from_dict({
        "nodes": {
            "loader": {"type": "data/loader"},
            "detector": {"type": "signals/detector"},
            "strategy": {"type": "strategy"},
        }
    })

# Warnings show auto-connections:
# UserWarning: Auto-connected 'loader' → 'detector' (data: ohlcv)
# UserWarning: Auto-connected 'detector' → 'strategy' (data: signals)
```

### Training-Only Detectors

Mark detectors that are only used for validator training:

```yaml
nodes:
  trend_detector:
    type: signals/detector
    name: sma_cross

  momentum_detector:
    type: signals/detector
    name: rsi_detector
    training_only: true  # Not passed to strategy

  validator:
    type: signals/validator
    # Receives signals from both detectors for training

  strategy:
    type: strategy
    # Only receives signals from trend_detector
```

### Signal Priority

Strategy nodes prefer `validated_signals` over raw `signals`:

```python
dag = FlowDAG.from_dict({
    "nodes": {
        "loader": {"type": "data/loader"},
        "detector": {"type": "signals/detector"},
        "validator": {"type": "signals/validator"},
        "strategy": {"type": "strategy"},
    }
})

# strategy receives validated_signals from validator
# (not raw signals from detector)
```

---

## StrategySubgraph

The strategy node is a composite node with internal DAG structure.

### Structure

```
┌──────────────────────────────────────────────────────────────┐
│                    STRATEGY (Composite Node)                  │
│                                                               │
│  inputs: [ohlcv, signals/validated_signals]                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              SIGNAL RECONCILIATION                       │ │
│  │   mode: any | all | weighted | model                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    ENTRY LAYER                           │ │
│  │   entry_mode: sequential | parallel | voting            │ │
│  │   entry_rules → entry_filters → position_manager        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              STRATEGY MODEL (optional)                   │ │
│  │   strategy_model + fallback_entry + fallback_exit       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                     EXIT LAYER                           │ │
│  │   exit_rules (parallel) → exit_merger → runner          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   METRICS LAYER                          │ │
│  │   metrics (parallel, independent)                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  outputs: [trades, metrics]                                  │
└──────────────────────────────────────────────────────────────┘
```

### StrategySubgraph

```python
@dataclass
class StrategySubgraph:
    # Entry configuration
    entry_rules: list[dict]
    entry_filters: list[dict]
    entry_mode: EntryMode = EntryMode.SEQUENTIAL

    # Exit configuration
    exit_rules: list[dict]

    # Model configuration
    strategy_model: dict | None = None
    fallback_entry: dict | None = None
    fallback_exit: dict | None = None

    # Signal reconciliation
    signal_reconciliation: SignalReconciliation = SignalReconciliation.ANY

    # Metrics
    metrics: list[dict]
```

### Enums

```python
class EntryMode(str, Enum):
    SEQUENTIAL = "sequential"  # Check in order, first match wins
    PARALLEL = "parallel"      # Check all, reconcile results
    VOTING = "voting"          # All vote, majority wins

class SignalReconciliation(str, Enum):
    ANY = "any"           # Any signal triggers entry
    ALL = "all"           # All signals must agree
    WEIGHTED = "weighted" # Weighted voting
    MODEL = "model"       # Model decides
```

### Example

```python
from signalflow.config import FlowDAG, StrategySubgraph

dag = FlowDAG.from_dict(config)
strategy = dag.get_strategy_node()
subgraph = StrategySubgraph.from_node(strategy)

print(subgraph.entry_mode)  # EntryMode.SEQUENTIAL
print(subgraph.signal_reconciliation)  # SignalReconciliation.ANY

# Get internal edges for visualization
edges = subgraph.get_internal_edges()
# [('signal_reconciler', 'entry_dispatcher', 'reconciled_signals'), ...]
```

---

## Complete YAML Example

```yaml
# Complex flow with multiple data sources, detectors, validator
id: ml_grid_strategy
name: "ML-Validated Grid Strategy"

nodes:
  # Multiple data loaders
  binance_loader:
    type: data/loader
    name: binance/spot
    store:
      path: data/binance.duckdb
    config:
      pairs: [BTCUSDT, ETHUSDT]
      timeframe: 1h

  bybit_loader:
    type: data/loader
    name: bybit/futures
    store:
      path: data/bybit.duckdb
    config:
      pairs: [BTCUSDT]

  # Features
  sma_features:
    type: feature/group
    config:
      features:
        - name: sma
          params: { period: 20 }
        - name: sma
          params: { period: 50 }

  # Detectors
  trend_detector:
    type: signals/detector
    name: example/sma_cross
    config:
      fast_period: 20
      slow_period: 50

  momentum_detector:
    type: signals/detector
    name: rsi_detector
    training_only: true  # Only for validator training

  # Validator
  ml_validator:
    type: signals/validator
    name: lightgbm

  # Strategy
  strategy:
    type: strategy
    config:
      signal_reconciliation: any

      entry_rules:
        - type: signal
          base_position_size: 200
        - type: momentum
          min_momentum: 0.5

      entry_mode: sequential

      entry_filters:
        - type: price_distance_filter
          min_distance_pct: 0.02

      exit_rules:
        - type: tp_sl
          take_profit_pct: 0.02
          stop_loss_pct: 0.01
        - type: trailing_stop
          trail_pct: 0.015

      # Optional ML model
      strategy_model:
        type: lightgbm_strategy
        model_path: models/strategy.pkl
      fallback_entry:
        type: signal
      fallback_exit:
        type: tp_sl

      metrics:
        - type: sharpe_ratio
        - type: win_rate
        - type: max_drawdown

config:
  capital: 50000
  fee: 0.001
```

---

## Integration with sf-kedro

```python
# sf-kedro wraps signalflow.config with default paths
from sf_kedro.utils.flow_config import load_flow_config

# Uses sf-kedro/conf/base as default path
config = load_flow_config("grid_sma")
```

## Integration with sf-ui

```python
# sf-ui uses FlowDAG for graph-based flows
from signalflow.config import FlowDAG

# Frontend sends FlowGraph JSON → backend converts to FlowDAG
dag = FlowDAG.from_dict(frontend_graph)

# Validate and get execution plan
errors = dag.validate()
if not errors:
    plan = dag.get_execution_plan()
```

---

## See Also

- [Builder API](api.md) - `sf.Backtest.from_dict()`
- [Strategy](strategy.md) - Entry/exit rules
- [Detector](detector.md) - Signal detection
- [Validator](validator.md) - Signal validation
