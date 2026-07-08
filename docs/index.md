---
title: Home
description: >
  SignalFlow is a Polars-backed Python framework for trading signal research
  and execution - one deployable object from idea to backtest to live.

hide:
  - navigation
---

# SignalFlow

> Current stable version: **{{ project_version }}**

SignalFlow is a Polars-backed framework for algorithmic trading research that
takes a strategy from idea to backtest to live with **one object you can save
and ship**. The public surface is six nouns.

---

## The six nouns

| Noun | What it is |
|------|------------|
| **Dataset** | One lazy, immutable market-data container. `sf.data(...)` builds it; the same object feeds backtest, paper, and live. |
| **Transform** | A column-producing step - features (`SMA` in core; `RSI`, `ATR`, `ZScore`, ... via the `signalflow-ta` plugin) and detectors (`SmaCrossDetector`, `ThresholdDetector`) share one contract. |
| **Models** | `ForecastModel` (trainable predictor -> probability column) plus validator combinators. |
| **Flow** | The central, deployable, tradeable unit: forecasts -> detectors -> validator -> strategy -> risk. |
| **Engine** | The decision/execution loop and brokers (`SimBroker` for backtest/paper, `BinanceBroker` for armed live). |
| **Run** | The result of executing a Flow - equity curve, fills, and a standard `.scorecard()`. |

See the [Concepts](concepts.md) page for the tier stack and the invariants, and
the [Glossary](glossary.md) for per-term definitions.

---

## Idea -> first backtest

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

The [Quick Start](quickstart.md) walks through this same example plus the
save/load round-trip.

---

## Three invariants worth knowing

**Leak-free OOS by mechanism.** A `ForecastModel` trains out-of-fold with an
embargo, and `predict_oos` returns values only inside the trained out-of-sample
span. A `Provenance` stamp and the `LeakageError` guard make in-sample scoring a
raised exception, not a silent bias.

**Backtest == simulate.** `flow.backtest` precomputes signals over a finished
Dataset; `flow.simulate` replays the identical loop used live, one bar at a time.
When a flow is causal the two agree exactly, so a mismatch is a look-ahead bug you
can catch before deploying.

**Deploy is data.** `flow.save(path, model_dir=...)` serializes the whole stack
(config plus trained artifacts) to YAML and a model directory; `sf.Flow.load(path)`
brings it back to a byte-identical backtest. Promoting a strategy is moving a file.

---

## Backtest -> paper -> live

One decision core drives all three modes. Backtest and paper replay a finished
Dataset; live consumes a streaming feed and routes orders to a real venue when
`armed=True`.

```python
flow.paper(ds, capital=50_000)                         # sim fills over a Dataset

feed = sf.PollingFeed(sf.BinanceSource(), pairs=["BTCUSDT"], interval="1m")
flow.live(feed, capital=50_000)                        # live data, SimBroker (paper)
```

See [Live & Walk-Forward](guide/live-and-walk-forward.md) for the walk-forward
and rolling-refit workflow.

---

## SignalFlow ecosystem

<div class="grid cards" markdown>

-   :material-package-variant-closed:{ .lg .middle } **signalflow-trading** (Core)

    ---

    Dataset, Transform, ForecastModel, Flow, Engine, Run. Polars-first, deploy-is-data.

    ```bash
    pip install signalflow-trading
    ```

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[signalflow-ta](ecosystem/signalflow-ta.md)**

    ---

    248 technical-indicator features + 21 detectors, physics-based market analogs.

    ```bash
    pip install "signalflow-trading[ta]"
    ```

-   :material-brain:{ .lg .middle } **[signalflow-labs](ecosystem/signalflow-labs.md)**

    ---

    Neural encoders (LSTM, Transformer, PatchTST, TCN) and an RL strategy.

    ```bash
    pip install "signalflow-trading[labs]"
    ```

</div>

---

## Getting started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **[Installation](getting-started/installation.md)**

    ---

    Install SignalFlow and verify the registry.

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](quickstart.md)**

    ---

    Build and round-trip your first Flow.

-   :material-sitemap:{ .lg .middle } **[Concepts](concepts.md)**

    ---

    The tier stack, the invariants, and where they are enforced.

-   :material-code-braces:{ .lg .middle } **[API Reference](api/index.md)**

    ---

    Full documentation for every public class and method.

</div>

---

## License

SignalFlow is released under the [MIT License](https://opensource.org/licenses/MIT).

!!! warning "Disclaimer"
    SignalFlow is provided for research purposes. Trading financial instruments
    carries risk. Past performance does not guarantee future results. Use at your
    own risk.
