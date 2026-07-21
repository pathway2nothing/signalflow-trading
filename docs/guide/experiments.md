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
  source: memory
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
  encode: default        # `default` = WoE + IVSelector; `null` = raw features
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
- `data` is forwarded to `sf.data(**data)` (so `cache_dir` caches fetched bars).
- `model.target` is built from the TARGET registry; `model.features` are built from
  the TRANSFORM registry into a `FeaturePipe`; `encode: default` uses WoE + IVSelector,
  `null` uses raw features.
- `scheme.walk_forward` runs `sf.walk_forward` and evaluates each of `metrics` per
  fold; `scheme: {fit: {}}` fits once and reports `classification_scorecard`.
- `backtest` (optional) fits the template on the full span, assembles a `Flow`, and
  runs `backtest(capital, oos=...)`.
- `tracking.mlflow` (an experiment name) logs the params and scorecard metrics.

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
