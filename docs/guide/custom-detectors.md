# Custom Detectors

A **detector** turns feature/forecast columns into a discrete `signal`
(`RISE` / `FALL` / `NONE`). It is a `Transform` whose single output column is
`signal`, so it composes with everything else and serializes into `flow.yaml`.

## The contract

```python
from dataclasses import dataclass

import polars as pl
import signalflow as sf
from signalflow.detector.base import SignalDetector
from signalflow.enums import NONE, RISE, SIGNAL_COL


@sf.detector("sma_above")
@dataclass
class SmaAboveDetector(SignalDetector):
    """RISE when close is above its SMA."""

    length: int = 50

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        sma = pl.col("close").rolling_mean(self.length).over("pair")
        rise = pl.col("close") > sma
        return df.with_columns(
            pl.when(rise).then(pl.lit(RISE)).otherwise(pl.lit(NONE)).alias(SIGNAL_COL)
        )
```

Rules that matter:

- **Decorate with `@sf.detector("name")` above `@dataclass`** - the name is what
  `flow.yaml` serializes and the registry reconstructs.
- **Keep parameters flat scalars** (ints/floats/strings/bools). Flat params
  round-trip through `flow.yaml` via the registry; nested objects do not.
- **Be causal.** Use `.over("pair")` on rolling windows so a value at bar `t`
  never depends on the future and never crosses a pair boundary. `compute()`
  sorts by `(pair, ts)` before calling `detect()`.
- Emit `RISE` / `FALL` / `NONE` (from `signalflow.enums`) into the `signal`
  column; downstream a strategy acts only on non-`NONE` rows.

## Worked example: a reversal-probability detector

A common mean-reversion entry: go long when a **posterior reversal probability**
spikes, optionally vetoed by a turbulent volatility regime and gated to below an
SMA. The probability and the regime are shipped indicators in `signalflow-ta`
(`PosteriorReversalProb`, `GMMVolRegime3State`), so the detector just computes
them and thresholds - no model training required.

```python
from dataclasses import dataclass

import polars as pl
import signalflow as sf
from signalflow.detector.base import SignalDetector
from signalflow.enums import NONE, RISE, SIGNAL_COL
from signalflow.ta.probabilistic import GMMVolRegime3State, PosteriorReversalProb


@sf.detector("reversal_prob")
@dataclass
class ReversalProbDetector(SignalDetector):
    """RISE when posterior reversal probability is high (optional turb/SMA gates)."""

    z_window: int = 240
    stretch_threshold: float = 2.5
    base_rate: float = 0.05
    likelihood_strength: float = 5.0
    p_min: float = 0.95
    require_low_turb: bool = False
    turb_window: int = 1440
    turb_smoother: int = 60
    turb_max: float = 0.30
    require_below_sma: bool = False
    sma_window: int = 240

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        pr = PosteriorReversalProb(
            z_window=self.z_window,
            stretch_threshold=self.stretch_threshold,
            base_rate=self.base_rate,
            likelihood_strength=self.likelihood_strength,
        )
        out = pr.compute(df)
        cond = pl.col(pr.outputs[0]) > self.p_min
        if self.require_low_turb:
            gmm = GMMVolRegime3State(window=self.turb_window, smoother=self.turb_smoother)
            out = gmm.compute(out)
            cond = cond & (pl.col(gmm.outputs[-1]) < self.turb_max)
        if self.require_below_sma:
            out = out.with_columns(pl.col("close").rolling_mean(self.sma_window).over("pair").alias("_sma"))
            cond = cond & (pl.col("close") < pl.col("_sma"))
        return out.with_columns(
            pl.when(cond.fill_null(False)).then(pl.lit(RISE)).otherwise(pl.lit(NONE)).alias(SIGNAL_COL)
        )
```

The same class with different flags is a family of strategies: thresholding the
probability alone trades the most; adding the turbulence veto and the SMA gate
trades progressively less. Because every parameter is a flat scalar, each variant
is fully captured by its `flow.yaml`.

## Putting it in a Flow

```python
ds = sf.data("binance", pairs=["BTCUSDT"], start="2024-01-01", interval="1m")

flow = sf.Flow(
    name="reversal",
    detectors=[ReversalProbDetector(p_min=0.95)],
    strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.3), exit=sf.Exit(tp=0.03, sl=0.02)),
)

run = flow.backtest(ds, capital=10_000)
print(run.scorecard())
```

A detector that computes its own features (like the one above) needs no
`forecasts` slot - the signal is derived directly from the data.

## It survives serialization and runs live unchanged

Because the detector is registered and its params are flat, the whole flow is
data:

```python
flow.save("flows/reversal/flow.yaml")
same = sf.Flow.load("flows/reversal/flow.yaml")          # detector rebuilt from the registry
```

And the signal is causal, so the vectorized backtest and the incremental
walk-forward (`flow.simulate`, the same loop that drives live) agree bar for bar:

```python
bt = flow.backtest(ds, capital=10_000)
sim = flow.simulate(ds, capital=10_000)                  # recompute per bar, only past data
assert sim.final_equity == bt.final_equity               # no look-ahead
```

See [Live & Walk-Forward](live-and-walk-forward.md) for the live loop and the
walk-forward refit/cache, and [Custom Data Types](custom-data-types.md) for
sources.
