# Quick Start

Build your first Flow, back it, and round-trip it to disk. Everything here runs
offline against the built-in `memory` source - no API keys.

!!! tip "New to the vocabulary?"
    The [Concepts](concepts.md) page explains the tier stack and the invariants;
    the [Glossary](glossary.md) defines each term.

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

What each step does:

1. `sf.data("memory", ...)` builds a **Dataset** - one lazy, immutable container.
2. `sf.ForecastModel(...)` pairs a **Target** (`FixedHorizon`) with a **FeaturePipe**;
   `fit` trains it out-of-fold.
3. `sf.Flow(...)` assembles the deployable stack: forecasts -> detectors -> strategy.
4. `flow.backtest(...)` returns a **Run**; `run.scorecard()` is the standard metric dict.

---

## Deploy is data

`flow.save` serializes the whole stack - config plus trained artifacts - to YAML
plus a model directory. `sf.Flow.load` restores a byte-identical backtest, so
promoting a strategy is moving a file.

```python
flow.save("flows/rsi_rise.yaml", model_dir="flows/models")   # yaml + trained artifacts
same = sf.Flow.load("flows/rsi_rise.yaml")
assert same.backtest(ds, capital=50_000).final_equity == run.final_equity
```

Model artifacts can live on the local filesystem, MLflow, or the Hugging Face Hub
(`model.save("mlflow://...")`, `model.save("hf://...")`).

---

## Backtest -> paper -> live

One decision core drives all three modes:

```python
flow.paper(ds, capital=50_000)                         # sim fills over a Dataset

feed = sf.PollingFeed(sf.BinanceSource(), pairs=["BTCUSDT"], interval="1m")
flow.live(feed, capital=50_000)                        # live data, SimBroker (paper)
```

---

## Next steps

<div class="grid cards" markdown>

-   :material-sitemap:{ .lg .middle } **[Concepts](concepts.md)**

    ---

    The tier stack, the leakage invariant, and warmup semantics.

-   :material-chart-timeline-variant:{ .lg .middle } **[Live & Walk-Forward](guide/live-and-walk-forward.md)**

    ---

    Walk-forward evaluation and rolling WoE refit.

-   :material-code-braces:{ .lg .middle } **[API Reference](api/index.md)**

    ---

    Every public class and method.

</div>
