# API Reference

Reference documentation for the public SignalFlow API. Every symbol below is
re-exported from the top-level `signalflow` package, so it can be imported
directly:

```python
import signalflow as sf

ds = sf.Dataset(...)
flow = sf.Flow(...)
```

## Modules

- [Data](data.md) - `Dataset`, the `data` loader, and market data sources
- [Transform & Features](transform.md) - feature transforms, pipelines, and selection
- [Target](target.md) - labeling specs and sample selection
- [Models](models.md) - forecast models and validator combinators
- [Detector](detector.md) - signal detectors
- [Engine](engine.md) - execution engine, brokers, orders, and fills
- [Strategy](strategy.md) - rules-based strategies and decision rules
- [Flow & Live](flow.md) - the `Flow` object, runs, and live feeds
- [Experiment](experiment.md) - experiments, scorecards, and statistics
- [CLI](cli.md) - command-line interface
- [Technical Analysis (ta)](ta.md) - indicators from signalflow-ta

## Enums and registry

The package also exposes shared enums and the component registry:

::: signalflow.registry
    options:
      show_root_heading: true

::: signalflow.ComponentType
    options:
      show_root_heading: true

::: signalflow.Signal
    options:
      show_root_heading: true

::: signalflow.RunMode
    options:
      show_root_heading: true

::: signalflow.Provenance
    options:
      show_root_heading: true

::: signalflow.Side
    options:
      show_root_heading: true

::: signalflow.IntentKind
    options:
      show_root_heading: true
