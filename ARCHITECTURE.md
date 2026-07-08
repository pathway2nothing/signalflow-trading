# SignalFlow architecture

A durable, implementation-synced map of `signalflow-trading`. It describes the code
as it is, not a proposal. Every path below resolves under `src/signalflow/`.

## The one-object idea

A strategy is one `Flow`: a declarative stack of `forecasts -> detectors ->
validator -> strategy -> risk`. The same `Flow` object runs `backtest`, `paper`,
and `live`, and serializes to a YAML file plus a model directory. Research,
promotion, and trading all move the same artifact.

## Module map

| Module | Responsibility |
|--------|----------------|
| `data/` | `Dataset` (`data/dataset.py`) - lazy, immutable market-data container; sources under `data/source/` (`memory.py`, `binance.py`, `cached.py`). |
| `transform/` | `Transform` base and `FeaturePipe` (`transform/pipe.py`), the `SMA` feature, WoE/IV encoders (`transform/encode/`), and `build_pipe` (`transform/build.py`). |
| `target/` | `Target` base (`target/base.py`) and the labelers (`FixedHorizon`, `TripleBarrier`, and the rest of `target/*.py`). |
| `model/` | `ForecastModel` (`model/forecast.py`), walk-forward (`model/walkforward.py`), OOS/fold helpers (`model/oos.py`), validator combinators (`model/validators.py`), persistence (`model/store/`). |
| `sampler/` | `Sampler` base (`sampler/base.py`) and samplers (uniform, CUSUM, meta-labeling, uniqueness). |
| `detector/` | `SignalDetector` base (`detector/base.py`) and fusion (`detector/fusion.py`) - non-learned signal rules. |
| `strategy/` | `StrategyModel` (`strategy/base.py`), `RulesStrategy` (`strategy/rules.py`), `LLMStrategy` (`strategy/llm.py`), `Risk` (`strategy/risk.py`), `Observation` (`strategy/observation.py`). |
| `engine/` | `Engine`, brokers (`SimBroker`, `BinanceBroker`), clock, and order/fill types. |
| `flow/` | `Flow` (`flow/flow.py`), the shared decision loop (`flow/loop.py`), the live loop (`flow/live.py`), `Run` (`flow/run.py`), YAML serialization (`flow/yaml.py`). |
| `experiment/` | `ArtifactCache` (`experiment/cache.py`) - keyed fold cache. |
| `registry.py` | The seven-type component registry with lazy autodiscovery and entry-point plugins. |
| `enums.py` | `ComponentType`, `Provenance`, `RunMode`, and the signal constants. |
| `errors.py` | The exception hierarchy. |
| `cli/` | The `sf` command-line entry point. |
| `_hash.py` | `stable_hash` and `code_fingerprint` for cache keys and drift detection. |

## Invariants and where they are enforced

### Leak-free out-of-sample by mechanism
- `ForecastModel.fit` (`model/forecast.py`) trains on embargoed walk-forward folds:
  the embargo is `horizon_bars` wide, and each fold trains strictly before
  `test_start - embargo`. The stitched out-of-fold predictions are stored on
  `oos_`.
- `ForecastModel.predict_oos` (`model/forecast.py`) returns values only inside the
  cached OOS span; rows outside it are null, and `strict=True` raises
  `FingerprintMismatch`.
- `Sampler._require_oos` (`sampler/base.py`) raises `LeakageError` when a training
  set is built from in-sample forecasts instead of `oos=True` predictions.
- The shared loop's `enriched_signals` (`flow/loop.py`) routes forecasts and the
  validator through `predict_oos` when `oos=True`, so a leak-free backtest never
  fires on rows a model was trained on.

### backtest == simulate (no look-ahead)
- `Flow.backtest` runs the vectorized event loop (`run_event_loop` in
  `flow/loop.py`); `Flow.simulate` replays the identical decision core one bar at a
  time through the live loop (`run_live_loop` in `flow/live.py`).
- For a causal flow the two produce equal equity and fills. A mismatch localizes a
  look-ahead bug. `Flow.simulate` reserves a leading `warmup` window (default
  `required_warmup`) that fills buffers without trading.

### A Flow is inference-only
- `Flow._check_fitted` (`flow/flow.py`) raises `UntrainedModelError` if any forecast
  or validator slot holds an unfitted model.
- `Flow._check_wiring` / `_check_targets` (`flow/flow.py`) raise `FlowConfigError`
  when a detector references a missing slot or a wrong target type.

### Deploy is data
- `Flow.save` / `Flow.load` delegate to `save_flow` / `load_flow` (`flow/yaml.py`),
  writing declarative component configs plus pinned model URIs.
- Model artifacts are written by `model/store/_layout.py`: `model.pkl`
  (cloudpickle of the fitted model), `oos/predictions.parquet`,
  `oos/fingerprint.json`, and `signature.json`. Loading re-attaches the parquet OOS
  as the authoritative copy.

### Warmup determinism
- `Flow.required_warmup` (`flow/flow.py`) is the max over detector warmups and
  forecast/validator feature-pipe warmups. Fixing the warmup makes backtest and
  live cold-start cut the identical leading slice.

## Registry and plugins

`registry.py` holds a `ComponentType -> name -> ComponentInfo` map with seven types
(`SOURCE`, `TRANSFORM`, `MODEL`, `STRATEGY`, `SAMPLER`, `BROKER`, `METRIC`) plus a
`TARGET` type used by targets. Discovery is lazy: `autodiscover` walks the
`signalflow.*` packages and then loads `signalflow.components` entry points, so an
installed plugin (`signalflow-ta`, `signalflow-labs`) registers its components with
no imports or wiring. A registered name is exactly what `flow.yaml` serializes and
`sf list` enumerates.

## Persistence story

Three distinct layers, deliberately separated:

1. **Declarative config** - transforms, detectors, strategy, and risk expose
   `to_config` / `from_config`, so the wiring round-trips through `flow.yaml`
   (`flow/yaml.py`) as plain data.
2. **Fitted state** - a trained `ForecastModel` is cloudpickled to `model.pkl`
   (`model/store/_layout.py`); the flow YAML references it by URI (filesystem,
   `mlflow://`, or `hf://`).
3. **OOS artifacts + fingerprints** - `oos/predictions.parquet` carries the
   leak-free predictions; `fingerprint.json` and `signature.json` capture the
   config/code/dataset identity (built from `_hash.py`) so drift is detectable and
   the fold cache (`experiment/cache.py`) can reuse unchanged folds.

## CLI

`cli/main.py` exposes `sf` subcommands: `sf list` (registered components), `sf info`
(a component's schema - description, role, module, parameters), `sf run` (load a flow
YAML, build a dataset, backtest, print the scorecard), `sf promote` (validate a flow
YAML and show the registry op), and `sf version`.
