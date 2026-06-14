# API Reference

Complete API documentation for SignalFlow components.

## Modules

- [High-Level API](api.md) - `Backtest`, `FlowBuilder`, `FlowResult`, shortcuts, exceptions
- [Configuration](config.md) - Flow configuration, FlowDAG, YAML loading, ArtifactSchema
- [Core](core.md) - Data containers (`RawData`, `Signals`), registry, semantic decorators
- [Data](data.md) - Exchange loaders, OHLCV resampling, storage backends
- [Detector](detector.md) - Signal detection algorithms
- [Feature](feature.md) - Feature extraction, `FeaturePipeline`, `FeatureSpec`, `ModelFeaturesPipeline`, informativeness
- [Models](models.md) - Pinned forecast artefacts: `ModelRef`, `Resolver`/`MlflowResolver`, `ModelRegistry`/`CachingModelRegistry`
- [Labeler](labeler.md) - Triple Barrier, Fixed Horizon, Trend Scanning, Volatility
- [Validator](validator.md) - ML-based signal validation (sklearn, LightGBM, XGBoost)
- [Strategy](strategy.md) - Runners, brokers, entry/exit rules, sizing, state, monitoring
- [Analytic](analytic.md) - Monte Carlo, Bootstrap CI, significance tests (Numba)
- [Visualization](viz.md) - D3.js flow graph, Mermaid export
- [CLI](cli.md) - Command-line interface
- [Technical Analysis (ta)](ta.md) - 189+ indicators from signalflow-ta
