# Changelog

All notable changes to SignalFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Architecture refactor toward the project's parity / single-source-of-truth
invariants (see `REFACTOR_PLAN.md`).

### Added

- **Event log as source of truth** (`signalflow.core.eventlog`): `apply_fill`,
  `fold`, `replay_state`, `portfolios_match`, and `CashPolicy`. The portfolio is
  now formally `fold(events)`; `apply_fill` is the single home of fill→position
  logic. `StrategyStore.read_trades` (read side) and `StrategyStore.verify_snapshot`
  make the saved snapshot a replay-verifiable cache.
- **ParityCheck** (`signalflow.strategy.ParityCheck` / `ParitySpec`): research↔
  production parity as a first-class, CI-checkable component. Diffs the event
  sequences of two runs, localises the first divergence, and classifies each
  component as `exact` / `approximate` / `out_of_scope` — failing like a
  `feature_hash` mismatch when a component leaves its declared class.
- **Reconciliation port** (`signalflow.strategy.live.reconciliation`): a
  `Reconciler` protocol + `LogMergeReconciler` adapter that reconciles the
  internal trade log against the exchange trade log over the canonical model.
- **Pinned strategy/RL models** (§4.1): `ModelEntryRule` / `ModelExitRule` accept
  a `model_ref` + `registry` and resolve via `ModelRef` → `ModelRegistry`
  (floating `latest` forbidden), matching forecast-model delivery.
- **Warmup contract** (`signalflow.core.warmup`): `warmup_bars_of`,
  `required_warmup_bars`, `assert_warmup_consistency`; the realtime runner now
  derives its warmup window from component declarations instead of a constant.

### Changed

- Brokers no longer implement position math: `create_position` /
  `apply_fill_to_position` removed from `Broker`; the three backtest brokers
  collapse onto a shared `process_fills` parameterised by `CashPolicy`
  (`unlimited` = `track_cash=False`).
- `StrategyState` snapshot deserialization now reconstructs the canonical
  `Portfolio` / `Position` (previously left as dicts).

### Removed

- Write-only `StrategyStore.upsert_positions` and the point-in-time `positions`
  table (the trade log + replay supersede it).
- Global mutable `_global_hooks` singleton (`configure_hooks` / `get_hooks`) —
  incompatible with parity; pass `HooksManager` via explicit DI instead.

### Moved

- The orphaned `signalflow.strategy.state` (`StateManager` + Redis/DuckDB
  backends, with its duplicate `Position`) moved to the work-in-progress
  `signalflow.strategy.live.state`. `signalflow.strategy.Position` is now the
  canonical `signalflow.core.Position`.

## [0.5.0] - 2026-02-14

### Added

- **Fluent Builder API**: `sf.Backtest()` with method chaining for intuitive backtest configuration.
  Multi-component support with named data sources, detectors, validators, entries, and exits.
  Signal aggregation modes: merge, majority, weighted, unanimous, any, meta_labeling.
  Cross-referencing between components (e.g., link entries to specific detectors).

- **One-liner shortcuts**: `sf.backtest()` and `sf.load()` convenience functions for quick
  prototyping without full builder configuration.

- **BacktestResult**: Rich result container with `.summary()`, `.plot()`, `.metrics`,
  `.metrics_df`, and Jupyter HTML rendering via `_repr_html_()`.

- **CLI Interface**: Command-line tools for backtest management:
  - `sf init` -- scaffold a sample YAML configuration
  - `sf validate <config.yaml>` -- validate configuration without running
  - `sf run <config.yaml>` -- execute backtest from YAML with optional `--plot` and `-o` output
  - `sf list detectors|metrics|features|all` -- discover available components

- **YAML Configuration**: Full backtest configuration via YAML files with component
  registration, cross-referencing, and validation.

- **Pipeline Visualization**: Interactive D3.js HTML visualization of backtest pipelines.
  Mermaid diagram export for documentation. Local development server via `sf.viz.serve()`.
  Graph extractors for BacktestBuilder, FeaturePipeline, and multi-source data flows.

- **OHLCV Resampling**: Unified timeframe resampling module (`signalflow.data.resample`):
  - `resample_ohlcv()` -- resample OHLCV data between timeframes
  - `align_to_timeframe()` -- auto-detect source TF and resample
  - `detect_timeframe()` -- detect timeframe from timestamp intervals
  - `select_best_timeframe()` -- find best exchange-supported timeframe
  - `can_resample()` -- check if resampling between two TFs is possible
  - Support for 12 standard timeframes across 8 exchanges

- **Auto-resampling**: `RawDataFactory.from_stores()` and `from_duckdb_spot_store()` accept
  `target_timeframe` parameter for automatic resampling during data loading.

- **Funding Rate Detector**: `FundingRateDetector` for perpetual futures that detects long
  entry opportunities when funding rate transitions from sustained positive to negative.
  Registered as `"funding/rate_transition"`.

- **Custom Exception Hierarchy**: `SignalFlowError` base with specialized exceptions
  (`DetectorNotFoundError`, `MissingDataError`, `InvalidParameterError`, etc.)
  providing actionable error messages with fix suggestions.

- **Lazy imports**: `sf.Backtest`, `sf.BacktestBuilder`, `sf.BacktestResult`, `sf.backtest`,
  `sf.load`, and `sf.viz` are lazy-loaded to reduce import overhead.

- **New exchange data sources**: Deribit, Kraken, Hyperliquid, WhiteBIT (spot & futures),
  Bybit inverse futures. Total: 7 exchanges with 15 loaders.

- **Multi-source data access**: `RawData` supports hierarchical multi-source access.
  `RawDataLazy` container for efficient deferred data loading.

- **Aggregated Open Interest**: Features for total OI, OI change, and OI z-score
  across single and multi-source data.

- **Signal & strategy analytics**: Classification, profile, distribution, and result metrics
  for signals and strategy performance analysis.

### Changed

- Standardized signal type naming and generalized signal handling across detectors and
  strategy components.

## [0.4.1] - Previous release

See [GitHub releases](https://github.com/pathway2nothing/signalflow-trading/releases) for earlier versions.
