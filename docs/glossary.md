---
title: Glossary
description: SignalFlow V5 terminology
---

# Glossary

Plain-language definitions of the SignalFlow vocabulary. For how the pieces fit
together and where the invariants are enforced, see [Concepts](concepts.md).

---

## Core nouns

### Dataset
One lazy, immutable market-data container built by `sf.data(...)`. The same object
feeds `backtest`, `paper`, and `live` - there is no separate live data path to drift.

```python
import signalflow as sf
ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")
```

### Transform
A column-producing step over a Dataset. Features (e.g. `SMA`) and detectors
(e.g. `ThresholdDetector`) share one `Transform` contract, so both serialize the
same way and both appear in the registry under a name.

### FeaturePipe
An ordered group of feature transforms. It is fit/serializable on its own and is
what a `ForecastModel` computes its inputs from.

```python
pipe = sf.FeaturePipe(sf.SMA(10), sf.SMA(20))
```

### Target
The label definition a `ForecastModel` learns to predict - for example
`FixedHorizon(bars=12)` (was price higher after 12 bars) or `TripleBarrier(...)`.
Registered under the `TARGET` component type.

### ForecastModel
A trainable predictor that maps features to a probability column. `fit` trains it
out-of-fold with an embargo; the two prediction methods differ deliberately:

- **`predict`** - in-sample style prediction over any rows.
- **`predict_oos`** - returns values **only inside the trained out-of-sample span**;
  rows outside it are null. This is what keeps evaluation leak-free.

### SignalDetector
A deliberately simple, **non-learned** rule that turns a forecast (or raw columns)
into a discrete signal. `ThresholdDetector(forecast="rise", p_min=0.6)` fires when a
forecast column clears a threshold. Detectors are simple on purpose - the learning
lives in the `ForecastModel`, so the decision rule stays inspectable and reproducible.

### Validator slot
An optional second-opinion model on a `Flow`. It holds a trained `ForecastModel` (or
a combinator like `MeanValidator` / `MaxValidator` / `VoteValidator`) that can veto or
reweight signals before they reach the strategy.

### StrategyModel
The component that turns validated signals into intents (which pair, which side, how
much). `RulesStrategy` is the built-in rule-based strategy; `LLMStrategy` delegates to
an OpenAI-compatible server.

### Risk
The limit layer on a `Flow`: `max_drawdown`, `max_positions`, `max_notional_per_pair`,
and an optional kill-switch file. Intents that breach a limit are blocked.

### Engine
The decision/execution loop plus brokers. `SimBroker` fills simulated orders for
backtest and paper; `BinanceBroker` routes real orders when a live run is `armed`.

### Flow
The central deployable unit: `forecasts -> detectors -> validator -> strategy -> risk`.
Every forecast (and validator) slot must hold a **trained** model, or construction
raises `UntrainedModelError`. The same Flow object runs `backtest`, `paper`, and `live`.

### Run
The result of executing a Flow: equity curve, fills, and a standard `.scorecard()`
metric dict. `run.oos` flags an out-of-sample-only run.

### Provenance
The stamp recorded on model outputs that records which fold/span produced them. The
leakage guard reads it to raise `LeakageError` when a model is scored on data it was
trained on.

---

## Encoding & warmup

### WoE / IV
Weight-of-Evidence encoding is the default feature encoding: each feature is binned
and mapped to the log-odds of the target, monotone and leak-aware because it is fit
out-of-fold. **Information Value (IV)** scores each encoded feature; `IVSelector` keeps
only columns whose IV clears a threshold.

```python
from signalflow.transform.encode import WoE
model = sf.ForecastModel(
    target=sf.FixedHorizon(bars=12),
    features=sf.FeaturePipe(sf.SMA(10), sf.SMA(20)),
    encode=WoE(refit="1d", window="365d"),   # rolling refit on a trailing year
)
```

### Warmup
The minimum number of leading bars a feature needs before its output is stable. A
Flow derives `required_warmup` from its feature pipe; `simulate(warmup=N)` reserves a
leading window that fills buffers without trading. Fixing the warmup makes backtest and
live cold-start cut the identical slice, so parity holds.

---

## Backtest metrics

### Sharpe ratio
Annualized risk-adjusted return. Rule of thumb: `< 0` losing, `0-1` mediocre, `1-2`
good, `> 2` excellent. The annualization assumes bar-frequency returns; see
`Run.sharpe`.

### Max drawdown
Largest peak-to-trough decline over the run - the worst losing streak.

### Total return
Final equity over initial equity, minus one.

---

## Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| OHLCV | Open, High, Low, Close, Volume |
| OOS | Out-of-Sample (evaluation span) |
| IS | In-Sample (training span) |
| WoE | Weight of Evidence |
| IV | Information Value |
| SMA | Simple Moving Average |
| RSI | Relative Strength Index |
| ATR | Average True Range |
| DD | Drawdown |

---

## See also

- [Concepts](concepts.md) - the tier stack and the invariants
- [Quick Start](quickstart.md) - build your first Flow
- [API Reference](api/index.md) - class-level documentation
