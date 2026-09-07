# Declarative Experiments

An `experiment.yaml` runs a whole research loop - data, model, walk-forward (or a
single fit), metrics, an optional backtest, and optional MLflow tracking - from one
file. A run is reproducible from the yaml plus its seed alone.

Run it with `sf exp experiment.yaml` (or `run_experiment(path)` in Python).

## Vocabulary (v1)

Unknown top-level keys are rejected with `FlowConfigError`.

```yaml
kind: experiment
name: exp006
seed: 7
data:
  source: synthetic
  pairs: [BTCUSDT]
  start: "2023-01-01"
  end: "2023-06-01"
  interval: 1h
  cache_dir: null
model:
  backend: lightgbm
  output: p_rise
  target: {name: fixed_horizon, params: {bars: 12}}
  features:
    - {transform: sma, params: {length: 10}}
    - {transform: sma, params: {length: 50}}
    - {transform: woe}            # stateful steps are refitted inside every fold
    - {transform: iv_selector, params: {min_iv: 0.1}}
  cv: {scheme: rolling, step: 7d, window: 365d}   # or {scheme: kfold, n: 5}
scheme:
  walk_forward: {train: 90d, step: 30d}   # or `fit: {}` for a single fit
metrics: [auc, brier]
backtest:                # optional: fit on the full span, assemble a Flow, backtest
  capital: 10000
  oos: true
  slot: rise
  detectors:
    - {transform: threshold, params: {forecast: rise, p_min: 0.6}}
  strategy: {name: rules, params: {}}
tracking:
  mlflow: null           # an experiment name enables MLflow logging
```

Semantics:

- `seed` seeds `random`/`numpy` via `seed_everything`.
- `data` is forwarded to `sf.dataset(**data)` (so `cache_dir` caches fetched bars).
- `model.cv` selects the walk-forward scheme of `ForecastModel.fit` (`rolling` with
  `step`/`window`, or `kfold` with `n`); omitted means `Rolling("7d", "365d")`.
- `model.target` is built from the TARGET registry; `model.features` are built from
  the TRANSFORM registry into a `FeaturePipeline`; encoders (`woe`, `iv_selector`,
  `scaler`) are ordinary steps of that list and are refitted inside every fold.
- `scheme.walk_forward` runs `sf.walk_forward` and evaluates each of `metrics` per
  fold; `scheme: {fit: {}}` fits once and reports `classification_scorecard`.
- `backtest` (optional) fits the template on the full span, assembles a `Flow`, and
  runs `backtest(capital, oos=...)`.
- `tracking.mlflow` (an experiment name) logs the params and scorecard metrics, tags
  the run with its provenance (signalflow / ta / labs versions and editable-checkout
  commits, the working directory's commit with `+dirty` when modified, python,
  platform, polars, seed) and attaches the yaml as a `config/` artifact. The same
  tags come with every `experiment_run(...)`; `sf.provenance()` returns them as a dict.

## Tracking backends

Tracking is pluggable. `experiment_run(name, tracker=...)` takes a registered name, a
tracker instance, or a list to fan out; the built-ins import their package only when
used, so none is a required dependency:

| name | package | notes |
|---|---|---|
| `mlflow` | `mlflow` (`[live]` extra) | `tracking_uri=` option |
| `wandb` | `wandb` | experiment -> `project`; options go to `wandb.init` (e.g. `mode="offline"`) |
| `litlogger` | `litlogger` (Lightning AI) | options go to `litlogger.init(...)`: `root_dir`, `teamspace`, `print_url` |
| `null` | - | logs nothing |

```yaml
tracking:
  tracker: wandb            # or a list: [mlflow, litlogger]
  experiment: exp_007_targets
  options: {mode: offline}
```

`tracking: {mlflow: <experiment>}` remains the short form. Your own backend is a
class with `start / log_params / set_tags / log_metrics / log_artifact / end`
(subclass `sf.BaseTracker` and override what you support), registered with
`@sf.register_tracker("name")` or published under the `signalflow.trackers` entry
point by any package. A backend whose package is missing disables tracking with a
warning instead of failing the experiment.

### Model artifacts

`model.save(uri)` / `ForecastModel.load(uri)` accept `file://`, `mlflow://`,
`lit://` (Lightning AI, via `litmodels`) and `hf://`. On Lightning a bare name is
qualified with `LIGHTNING_TEAMSPACE` and the returned URI is pinned to the uploaded
version, so a fold model saved during a walk-forward is reproducible:

```python
result = sf.walk_forward(model, ds, train="6mo", step="1mo", save_to="lit://models/exp002_rise_{tag}")
fold = sf.ForecastModel.load("lit://models/exp002_rise_202401")
```

## Output

`run_experiment` returns and writes `results.json` next to the yaml:

```python
import signalflow as sf

result = sf.run_experiment("experiment.yaml")
print(result["folds"])            # per-fold metric rows (walk_forward scheme)
print(result["model_scorecard"])  # AUC / PR-AUC / Brier / precision / recall / F1
print(result["run_scorecard"])    # the backtest scorecard (when a backtest block is present)
```

Because everything - data span, features, target, seed, scheme - lives in the yaml,
the run reproduces exactly from the file: same seed, same fold scores.
