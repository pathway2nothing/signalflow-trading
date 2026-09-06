# Changelog

All notable changes to SignalFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking - the API is pre-1.0, no compatibility aliases are kept)

- `FeaturePipe` is now `FeaturePipeline` (`signalflow.transform.pipeline`); the
  registry name is `feature_pipeline` and the config role is `pipeline`. The
  constructor also accepts one list of transforms. `build_pipe()` became the
  classmethod `FeaturePipeline.from_names(...)`; `PipeError` is `PipelineError`.
  Saved `flow.yaml` files written with the old names must be regenerated.
- The deprecated source name `memory` / `MemorySource` and the registry's
  deprecated-name resolution were removed; use `synthetic` / `SyntheticSource`.
- One `Fold` dataclass (`signalflow.model.oos.Fold`, fields `train_start`,
  `train_end`, `test_start`, `test_end`, plus `model`/`oos` from `walk_forward`)
  replaces `WalkForwardFold` and the old `*_ts` field names.
- `signalflow._time` is the single home of `parse_duration`, `advance`/`retreat`
  (calendar months), `parse_datetime`/`to_epoch` (always UTC), `INTERVAL_SECONDS`
  and `bar_seconds`; the duplicate parsers in sources, the model and targets
  are gone. Epoch conversion of ISO dates is now explicitly UTC.
- `flow.loop` exports `EMPTY_SIGNALS_SCHEMA` and `orders_from_intents` (no
  underscore variants); `MIN_OOS_COVERAGE` lives only in `flow.bundle`;
  `client_order_id(order)` in `engine.types` replaces `sim_client_order_id` and
  `BinanceBroker.client_order_id`.
- `ForecastModel.predict` returns null wherever a raw feature input is null/NaN
  (the warmup window or an undefined value): training never sees such rows, so
  scoring them was an extrapolation from the encoder's missing bin. Detectors
  therefore no longer fire inside the warmup, and `flow.simulate()` with the
  default warmup now equals `flow.backtest()` for model flows too.

- The walk-forward scheme is a model parameter: `ForecastModel(cv=sf.Rolling(step,
  window))` (default `Rolling("7d", "365d")`) or `cv=sf.KFold(n)`; `n_folds` and
  the `WoE.refit`/`WoE.window` fields are gone. `WoE` is a pure encoding recipe
  refitted inside every fold. `experiment.yaml` takes `model.cv`. The old default
  (a daily refit) is `cv=sf.Rolling(step="1d", window="365d")`.

- Encoders are pipeline steps: `FeaturePipeline(SMA(10), WoE(), IVSelector())`.
  `ForecastModel` splits the pipeline at the first `requires_fit` transform, computes
  the stateless prefix once and refits a clone of the stateful tail inside every fold
  (`ForecastModel.tail_` holds the production tail). The `encode`/`select` fields are
  gone; a pipeline without a stateful step trains on raw features (logged at INFO).
  `WoE(replace=True)` drops the columns it encodes; `IVSelector` is a `narrows`
  transform (its outputs are the kept subset) and keeps every candidate, with a
  warning, when none clears `min_iv`. New `Scaler` (standard/robust, fit on train).
  `Transform` gained `clone()`, `is_fitted`, `removes` and `narrows`;
  `FeaturePipeline` gained `split()`. `experiment.yaml` rejects `model.encode`;
  list the encoder steps under `model.features`. `signature.json` now records
  `raw_columns` and `model_columns` instead of `encode`/`select_keep`.

- `Dataset.iter_bars` sorts once and yields zero-copy slices at precomputed offsets
  instead of iterating a `group_by`; `Bar` carries `high`/`low` maps next to `prices`
  so limit fills need no per-bar filter. The live/simulate loop keeps its trailing
  window as one frame with chunk appends and slice trims instead of re-concatenating
  the buffer every bar. Fills and equity are unchanged; the 357k-bar backtest that
  took 223 s runs in seconds (`SF_BENCH=1 pytest tests/test_bench.py`).

- `SimBroker(fill="next_open", filters={pair: {stepSize, tickSize, minQty,
  minNotional}})`: orders can execute at the next bar's open (what a loop deciding
  on a closed candle really gets) and are quantized with the same arithmetic as the
  venue (`engine.quantize`), so paper and armed runs size orders identically.
  `Bar` carries an `open` map. Default fills stay at the decision bar's close.
- Live loop: the state file is written atomically (temp file + rename) and only
  when the book changed; every fill is appended to `<state>.fills.jsonl`;
  `PollingFeed` retries transient source errors with backoff and counts the polls it
  had to skip; `late_bar_policy="skip"` refuses to trade a bar that arrived after
  `max_latency_s`; `Run.meta` reports skipped bars, feed errors and strategy fallbacks.
- `LLMStrategy` logs a WARNING and counts every fallback (`fallbacks`); with
  `fallback=None` a failed decision raises `KillSwitchTripped` instead of silently
  trading rules; the decision cache is bounded (`cache_size`). The client logs its
  own failures instead of swallowing them.
- `BinanceBroker`: `api_key`/`api_secret` are hidden from `repr`; every retried send
  is re-signed with a fresh `timestamp`; after a failed send the venue is queried by
  client-order id before the order is treated as unfilled.
- `Risk.clip` returns clipped copies instead of mutating the strategy's intents.

- `Labeler.horizon_field` names the forward look-ahead field explicitly (an int,
  a duration string, or a tuple of horizons); the attribute-name heuristic that
  could silently yield a one-bar embargo is gone, and a misdeclared field raises.
- `Run.promotable` defaults to `False` and a run is promotable only when it was
  scored with `oos=True` (and, for model flows, with enough OOS coverage) - a
  rule-only flow is in-sample too.
- Model artifacts carry `env.json` (python, signalflow and the numeric stack);
  loading under a different major.minor logs a warning. `hf://` artifacts unpickle
  remote code and are refused unless `ForecastModel.load(uri, trust_remote=True)` /
  `Flow.load(path, trust_remote=True)`.
- The registry imports an explicit list of core modules instead of walking every
  `signalflow.*` package, so registering plugins (`sf list`) no longer imports torch;
  `sf info` reports why a component could not be default-constructed instead of `n/a`.

### Fixed

- The fold cache (`ForecastModel.fit(cache=...)`) now folds the source code of
  every feature transform and of the target into its key, so editing a feature
  invalidates the cached out-of-fold predictions as the docs promised.
- `TripleBarrier` labels run through a numba kernel (python fallback when numba
  is absent) instead of a pure-python double loop; labels are unchanged.

### Added

- Step-level logging across the core: one INFO summary per `ForecastModel.fit`,
  `Flow.backtest`/`paper`, `Flow.simulate`/`live` and `walk_forward`, and DEBUG
  detail (enable with `SF_VERBOSE=1` / `SF_LOG_LEVEL=DEBUG`) for data loading,
  every `FeaturePipe` transform or fused feature group, fit internals (features,
  sampler, labels, each fold with its kept-column count, final stack),
  predictions, each forecast slot and detector in a run, the decision loop, live
  progress, the feature store and flow save/load; WoE/IV internals per fold at
  TRACE. `signalflow._logging.step` is the helper.
- All fixed-width Binance kline intervals (`1s 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h
  1d 3d 1w`) are accepted by `binance`, `synthetic`, the disk cache, and
  `PollingFeed`, from one table - `INTERVAL_SECONDS` / `interval_seconds()` in
  `data/source/base.py`. Calendar-month `1M` is rejected with a clear error.

### Changed

- Feature pipelines no longer re-sort and copy the frame per transform:
  `ensure_sorted` (in `transform/base.py`) checks (pair, ts) order and sorts only
  when needed, `FeaturePipe.compute` sorts once and chains consecutive expression
  features into one lazy query with a single `collect()`, detectors and the
  decision loop reuse the same check, and `Dataset.with_forecasts` appends
  predictions without a join when they carry the frame's own (pair, ts) index.
  The raw `Dataset.frame` is never modified.
- `flow.save(model_dir=...)` now calls `strategy.save_artifacts(model_dir)` when a
  strategy defines it, so strategies carrying trained artifacts (the labs
  `RLStrategy`) pin them like forecast models and round-trip through `Flow.load`.
- The built-in synthetic data source is now registered as `synthetic`
  (`sf.data("synthetic", ...)`, class `SyntheticSource`, file
  `data/source/synthetic.py`); `sf run --source` defaults to it. The old name
  `memory` / `MemorySource` still resolves with a deprecation warning and is
  no longer listed by `sf list source`.

## [0.8.5] - 2026-07-18

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
