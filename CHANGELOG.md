# Changelog

All notable changes to SignalFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `walk_forward()` / `WalkForwardResult` - anchored walk-forward evaluation that
  stitches per-fold out-of-sample runs into one equity curve.
- `build_pipe()` - build a `FeaturePipe` from a compact spec list.
- `CachedSource` and `sf.data(..., cache_dir=)` - transparent on-disk caching of
  fetched market data.
- `ForecastModel.operating_point` - store the chosen probability threshold on the
  trained model.
- `Flow.required_warmup` and `Flow.simulate(warmup=None)` - warmup is derived from
  the feature pipe when not given.
- `Flow.backtest(oos=...)` and `Run.oos` - run and flag out-of-sample-only backtests.
- `sf info` CLI command - print a registered component's schema (description, role,
  module, parameters).
- Duration-string target horizons (e.g. `FixedHorizon(bars="6h")`).
- `TARGET` registry component type.

### Changed

- `FlowConfigError` and `DegenerateTargetError` raised for misconfigured flows and
  degenerate label distributions.

## [0.8.4] - 2026-06-17

### Added

- Per-fold WoE state and out-of-sample predictions are cached across runs, so a
  repeated `ForecastModel.fit` / backtest on unchanged inputs skips refitting.

## [0.8.3] - 2026-06-16

### Added

- Declarative `to_config` / `from_config` for transforms and `FeaturePipe`, making
  the full feature stack reconstructable from a flow YAML.

### Fixed

- WoE encoders rebuild deterministically from config instead of carrying pickled state.

## [0.8.2] - 2026-06-15

### Added

- Real-time live loop (`Flow.live`) over a streaming feed and walk-forward
  simulation, sharing the single decision core with backtest and paper.

### Fixed

- Backtest fills and equity accounting corrected so `backtest == simulate`.

## [0.8.0] - 2026-06-14

Full V5 architecture rewrite. The public surface collapses to six nouns - Dataset,
Transform, Models, Flow, Engine, Run - and the fluent `sf.Backtest` builder is
removed.

### Added

- `Flow`: the deployable forecasts -> detectors -> validator -> strategy -> risk
  stack that runs `backtest`, `paper`, and `live` from one object.
- `ForecastModel` with embargoed out-of-fold training, `predict` vs `predict_oos`,
  and a leakage guard (`LeakageError`) enforced by Provenance stamps.
- Weight-of-Evidence / Information-Value feature encoding (`WoE`, `IVSelector`) as
  the default, fit out-of-fold.
- `Dataset` (`sf.data(...)`): one lazy, immutable market-data container feeding
  backtest, paper, and live.
- Deploy-is-data serialization: `Flow.save` writes YAML plus a model directory,
  `Flow.load` restores a byte-identical backtest. Artifacts on filesystem, MLflow,
  or the Hugging Face Hub.
- Seven-type component registry (SOURCE, TRANSFORM, MODEL, STRATEGY, SAMPLER,
  BROKER, METRIC) with entry-point plugin autodiscovery.

### Removed

- The V4 fluent builder (`sf.Backtest`, `BacktestBuilder`, `BacktestResult`),
  `RawDataFactory` / `VirtualDataProvider`, and `sf.viz`.

## [0.5.0] - 2026-02-14

### Added

- **Fluent Builder API**: `sf.Backtest()` with method chaining for backtest
  configuration, multi-component support, and signal aggregation modes.
- **CLI Interface**: `sf init`, `sf validate`, `sf run`, and `sf list`.
- **YAML Configuration**: full backtest configuration via YAML files.
- **OHLCV Resampling**: `signalflow.data.resample` across 12 timeframes and 8 exchanges.
- **New exchange data sources**: Deribit, Kraken, Hyperliquid, WhiteBIT, Bybit inverse.
- **Custom Exception Hierarchy**: `SignalFlowError` base with actionable messages.

## [0.4.1] - Previous release

See [GitHub releases](https://github.com/pathway2nothing/signalflow-trading/releases) for earlier versions.
