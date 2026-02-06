---
title: Ecosystem
---

# SignalFlow Ecosystem

SignalFlow is a modular ecosystem of Python packages for algorithmic trading.
Each package focuses on a specific domain while sharing the core framework's
component registry, data containers, and pipeline patterns.

---

## Packages

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **signalflow-trading** (Core)

    ---

    Core framework: data containers, signal detection, backtesting, strategy execution.
    Component registry, Polars-first processing, DuckDB storage.

    ```bash
    pip install signalflow-trading
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-trading){ .md-button }

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[signalflow-ta](signalflow-ta.md)** (Technical Analysis)

    ---

    199+ technical indicators across 8 modules: momentum, overlap, volatility,
    volume, trend, statistics, performance, divergence. Includes physics-based
    market analogs and preset pipeline factories.

    ```bash
    pip install signalflow-ta
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-ta){ .md-button }

-   :material-brain:{ .lg .middle } **[signalflow-nn](signalflow-nn.md)** (Neural Networks)

    ---

    Deep learning signal validators built on PyTorch Lightning. Encoder + Head
    composition pattern with LSTM, GRU encoders and MLP, Attention, Residual,
    Distribution, Ordinal, and Confidence heads.

    ```bash
    pip install signalflow-nn
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-nn){ .md-button }

</div>

---

## Architecture

All packages share the SignalFlow component registry via the `@sf_component` decorator.
This means indicators from signalflow-ta and validators from signalflow-nn are
automatically discoverable by the core framework:

```python
from signalflow.core import default_registry, SfComponentType

# After installing signalflow-ta, its indicators are available:
rsi_cls = default_registry.get(SfComponentType.FEATURE, "momentum/rsi")

# After installing signalflow-nn, its validators are available:
validator_cls = default_registry.get(SfComponentType.VALIDATOR, "temporal_validator")
```

### Dependency Chain

```
signalflow-trading          # Core (required)
├── signalflow-ta           # extends with 199+ indicators
└── signalflow-nn           # extends with neural network validators
```

Both extension packages depend on `signalflow-trading` and extend it
using Python namespace packages under `signalflow.*`.
