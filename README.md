<div align="center">

# SignalFlow

**Real-time-first framework for trading signal research and execution**

<p>
<a href="https://pypi.org/project/signalflow-trading/"><img src="https://img.shields.io/badge/version-1.0.0.dev0-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="License: MIT"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-1a1a2e" alt="Code style: ruff"></a>
</p>

</div>

---

SignalFlow V5 is a Polars-backed framework for algorithmic trading research that
takes a strategy from idea to backtest to live with **one object you can save and
ship**. The public surface is six nouns:

| Noun | What it is |
|------|------------|
| **Dataset** | One lazy, immutable market-data container. `sf.data(...)` builds it; the same object feeds backtest, paper, and live. |
| **Transform** | A column-producing step - features (`RSI`, `Returns`, `ZScore`) and detectors (`ThresholdDetector`) share one contract. |
| **Models** | `ForecastModel` (trainable predictor → probability column) plus validator combinators. |
| **Flow** | The central, deployable, tradeable unit: forecasts → detectors → validator → strategy → risk. |
| **Engine** | The decision/execution loop and brokers (`SimBroker`, `ExchangeBroker`). |
| **Run** | The result of executing a Flow - equity curve, fills, and a standard `.scorecard()`. |

## Install

```bash
pip install signalflow-trading
```

| Extra | Installs | For |
|-------|----------|-----|
| `signalflow-trading[ta]` | (signalflow-ta plugin) | 189+ technical indicators |
| `signalflow-trading[rl]` | stable-baselines3, gymnasium, torch | RL strategy models (`Flow.env`) |
| `signalflow-trading[live]` | mlflow, huggingface_hub | model artifact tracking / deploy |
| `signalflow-trading[llm]` | anthropic, pydantic | LLM-assisted flow authoring |
| `signalflow-trading[dev]` | pytest, ruff, mypy | development |

## Idea → first backtest

```python
import signalflow as sf

ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")

model = sf.ForecastModel(target=sf.FixedHorizon(bars=12),
                         features=sf.FeaturePipe(sf.SMA(10), sf.SMA(20), sf.SMA(50)))
model.fit(ds)                                          # train tier-1 forecaster

flow = sf.Flow(name="sma_rise",
               forecasts={"rise": model},
               detectors=[sf.ThresholdDetector(forecast="rise", p_min=0.6)],
               strategy=sf.RulesStrategy())
run = flow.backtest(ds, capital=50_000)
print(run.scorecard())                                 # total_return, sharpe, max_drawdown, ...
```

## CLI

```bash
sf list                 # registry snapshot grouped by type
sf list transform       # one type, with one-line summaries
sf run flow.yaml --source memory --pairs BTCUSDT --start 2023-01-01 --interval 1h --capital 50000
sf promote flow.yaml --to shadow   # validate + show the registry op (real promotion: sf-prod)
sf version
```

## Three invariants worth knowing

**WoE/IV encoding is the default.** Features flow through Weight-of-Evidence
encoding against the target, and `IVSelector` keeps only columns whose Information
Value clears a threshold. Encoding is monotone, leak-aware, and fit out-of-fold.

**A Flow is inference-only.** Every forecast slot (and the optional validator
slot) must hold a *trained* model. Constructing a `Flow` around an unfitted model
raises `UntrainedModelError` - you cannot accidentally deploy something untrained.
The same Flow object runs `backtest`, `paper`, and `live` over the same Dataset;
only the mode flag changes.

**Deploy is data.** `flow.save(path)` serializes the whole stack (config +
trained artifacts) to YAML plus a model directory; `sf.Flow.load(path)` brings it
back byte-for-byte. There is no code to redeploy - promoting a strategy is moving
a file. Model artifacts can live on the local filesystem, MLflow, or the Hugging
Face Hub (`model.save("mlflow://...")`, `model.save("hf://...")`).

```python
flow.save("flows/rsi_rise.yaml")
same = sf.Flow.load("flows/rsi_rise.yaml")
assert same.backtest(ds, capital=50_000).final_equity == run.final_equity
```

## Registry

Every core class registers under a name - that name is what `flow.yaml`
serializes and `sf list` enumerates. Seven `ComponentType`s: SOURCE, TRANSFORM,
MODEL, STRATEGY, SAMPLER, BROKER, METRIC.

```python
sf.registry.snapshot()                          # {type: [names]}
sf.registry.list(sf.ComponentType.TRANSFORM)    # ['atr', 'bbands', 'ema', ...]
```

## Ecosystem

| Package | Description |
|---------|-------------|
| **signalflow-ta** | Technical-indicator plugin (`[ta]` extra) |
| **sf-prod** | Promotion, shadow/live rollout, monitoring |

---

**License:** MIT &ensp;·&ensp; **Author:** [pathway2nothing](https://github.com/pathway2nothing) &ensp;·&ensp; **Docs:** [signalflow-trading.com](https://signalflow-trading.com)
