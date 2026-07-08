---
title: Ecosystem
---

# SignalFlow Ecosystem

SignalFlow is a modular ecosystem of Python packages for algorithmic trading.
Each package focuses on a specific domain while sharing the core framework's
component registry, data containers, and Flow patterns.

---

## Packages

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **signalflow-trading** `v0.8.4` (Core)

    ---

    Core framework: `Dataset`, `Transform`/`FeaturePipe`, `ForecastModel`, `Flow`,
    `Engine`, `Run`. Component registry, Polars-first processing, deploy-is-data
    YAML serialization.

    ```bash
    pip install signalflow-trading
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-trading){ .md-button }

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[signalflow-ta](signalflow-ta.md)** `v0.8.2`

    ---

    248 technical-indicator features across 8 modules (momentum, overlap, volatility,
    volume, trend, statistics, performance, divergence) plus 21 signal detectors.
    Physics-based market analogs and AutoFeatureNormalizer.

    ```bash
    pip install "signalflow-trading[ta]"
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-ta){ .md-button }

-   :material-brain:{ .lg .middle } **[signalflow-labs](signalflow-labs.md)** `v0.8.2`

    ---

    Neural encoders (LSTM, GRU, Transformer, PatchTST, TCN, TSMixer, InceptionTime),
    classification heads, and an RL strategy. Built on PyTorch.

    ```bash
    pip install "signalflow-trading[labs]"
    ```

    [:material-github: GitHub](https://github.com/pathway2nothing/signalflow-labs){ .md-button }

</div>

---

## Architecture

All packages share the SignalFlow component registry via semantic decorators
(`@sf.detector`, `@sf.feature`, `@sf.transform`, `@sf.model`, `@sf.strategy`).
Components from any installed package are discoverable through the same registry:

```python
import signalflow as sf
import signalflow.ta  # noqa: F401 - registers the ta components

rsi_cls = sf.registry.get(sf.ComponentType.TRANSFORM, "momentum/rsi")
sf.registry.snapshot()  # {type: [names]} across every installed package
```

Installing a plugin auto-registers its components via the `signalflow.components`
entry point - no imports or wiring needed.

### Dependency Chain

```
signalflow-trading              # Core (required)
├── signalflow-ta               # 248 features + 21 detectors
├── signalflow-labs             # Neural encoders, RL strategy
└── sf-custom                   # User components (entry-point autodiscovery)
```

Extension packages use Python namespace packages under `signalflow.*`.
Custom packages register through the `signalflow.components` entry point.
