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

-   :material-package-variant-closed:{ .lg .middle } **signalflow-trading** `v0.6.0` (Core)

    ---

    Core framework: data containers, signal detection, backtesting, strategy execution,
    state persistence, statistical analysis. Component registry, Polars-first processing, DuckDB storage.

    ```bash
    pip install signalflow-trading
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-trading){ .md-button }

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[signalflow-ta](signalflow-ta.md)** `v0.6.0`

    ---

    189+ technical indicators across 8 modules: momentum, overlap, volatility,
    volume, trend, statistics, performance, divergence. 24 signal detectors.
    Physics-based market analogs and AutoFeatureNormalizer.

    ```bash
    pip install signalflow-ta
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-ta){ .md-button }

-   :material-brain:{ .lg .middle } **[signalflow-nn](signalflow-nn.md)** `v0.6.0`

    ---

    14 neural encoders (LSTM, GRU, Transformer, PatchTST, TCN, TSMixer, InceptionTime),
    7 classification heads, 4 loss functions. Built on PyTorch Lightning.

    ```bash
    pip install signalflow-nn
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-nn){ .md-button }

-   :material-pipe:{ .lg .middle } **[sf-kedro](sf-kedro.md)** `v0.5.0`

    ---

    Universal ML pipelines: backtest, analyze, train, tune (Optuna), validate (walk-forward).
    Flow configuration via YAML. MLflow and Telegram integrations.

    [:material-github: GitHub](https://github.com/pathway2nothing/sf-kedro){ .md-button }


</div>

---

## Architecture

All packages share the SignalFlow component registry via semantic decorators
(`@sf.detector`, `@sf.feature`, `@sf.entry`, `@sf.exit`, etc.).
Components from any installed package are automatically discoverable:

```python
from signalflow.core import default_registry, SfComponentType

# signalflow-ta indicators
rsi_cls = default_registry.get(SfComponentType.FEATURE, "momentum/rsi")

# signalflow-nn validators
validator_cls = default_registry.get(SfComponentType.VALIDATOR, "temporal_validator")

# Custom components (via entry-point autodiscovery)
custom_cls = default_registry.get(SfComponentType.DETECTOR, "custom/my_detector")
```

### Dependency Chain

```
signalflow-trading              # Core (required)
├── signalflow-ta               # 189+ indicators
├── signalflow-nn               # Neural network encoders
├── sf-kedro                    # ML pipelines
└── sf-custom                   # User components (entry-point autodiscovery)
```

Extension packages use Python namespace packages under `signalflow.*`.
Custom packages use `signalflow.components` entry point for automatic registration.
