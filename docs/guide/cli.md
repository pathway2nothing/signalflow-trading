# CLI & YAML Config

The `sf` command line wraps the same package you import as `signalflow`. It has
five commands: `list`, `info`, `run`, `promote`, and `version`. Everything it
runs is registry-driven, so any component you register with `@sf.register_*`
shows up automatically.

Invoke it as `sf <command>` once installed, or `python -m signalflow.cli.main
<command>` from a checkout.

## `list` - enumerate the registry

List every registered component grouped by type, or pass a single type to get a
table with one-line summaries.

```console
$ sf list
broker (3)
  binance
  exchange
  sim

sampler (4)
  cusum
  meta_labeling
  uniform
  uniqueness
...
```

```console
$ sf list transform
             transform components (8)
name          summary
------------  --------------------------------------------------
feature_pipe  Run child transforms in order; outputs are ...
sma           Simple moving average of close.
sma_cross     RISE when the fast SMA crosses above the slow SMA.
threshold     RISE when a forecast's probability exceeds p_min.
...
```

The valid types are `source`, `transform`, `target`, `model`, `strategy`,
`sampler`, `broker`, and `metric`.

## `info` - inspect one component

Show a component's schema: description, role, module, its constructor parameters
(name, type, default, required), and the outputs/warmup of a default instance.

```console
$ sf info transform threshold
threshold (ThresholdDetector)
RISE when a forecast's probability exceeds ``p_min``.
role: detector
module: signalflow.detector.fusion

parameters
param     type / default
--------  -------------------------------------
forecast  str  default=''  required=False
p_min     float  default=0.6  required=False
output    str  default='p_rise'  required=False

outputs: n/a
warmup: n/a
```

## `run` - backtest a flow.yaml

Load a saved flow, build a dataset from a registered source, backtest, and print
the scorecard. Data options default to a small `memory` dataset for a quick smoke
test.

```console
$ sf run flow.yaml --source memory --pairs BTCUSDT --start 2023-01-01 --end 2023-03-01 --interval 1h --capital 10000
scorecard - sma_rise
metric          value
--------------  --------
name            sma_rise
mode            backtest
promotable      True
oos             False
n_fills         11
initial_equity  10000.0
final_equity    9797.93
total_return    -0.0202
max_drawdown    0.0336
sharpe          -2.412
```

Options: `--source`, `--pairs` (comma-separated), `--start`, `--end`,
`--interval`, `--capital`.

## `promote` - validate and show the promotion op

Load and validate a flow, then report the registry operation promotion would
perform. The real promotion into sf-prod happens there, not here.

```console
$ sf promote flow.yaml --to shadow
validated flow 'sma_rise' from flow.yaml
would register: stage=shadow flow='sma_rise' quote=USDT
no server contacted - real promotion happens in sf-prod.
```

## `version` - print the installed version

```console
$ sf version
0.8.5
```

## Where flow.yaml comes from

A `flow.yaml` is what `Flow.save(path)` writes: the flow name, quote, forecast
model URIs, detector configs, the strategy, and risk limits. `Flow.load(path)`
rebuilds a byte-identical flow. See [Custom Detectors](custom-detectors.md) for
how a registered detector round-trips through it.
