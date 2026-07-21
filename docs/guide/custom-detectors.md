# Custom Detectors

A **detector** turns feature/forecast columns into a discrete `signal`
(`RISE` / `FALL` / `NONE`). It is a `Transform` whose single output column is
`signal`, so it composes with everything else and serializes into `flow.yaml`.

## The contract

```python
from dataclasses import dataclass

import polars as pl
import signalflow as sf


@sf.register_detector("sma_above")
@dataclass
class SmaAbove(sf.SignalDetector):
    """RISE while close is above its n-bar SMA."""

    n: int = 20

    @property
    def warmup(self) -> int:
        return self.n

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        sma = pl.col("close").rolling_mean(self.n).over("pair")
        rise = pl.col("close") > sma
        return df.with_columns(pl.when(rise).then(pl.lit(sf.RISE)).otherwise(pl.lit(sf.NONE)).alias("signal"))
```

Rules that matter:

- **Register with `@sf.register_detector("name")` above `@dataclass`** - the name
  is what `flow.yaml` serializes and the registry reconstructs. The class **must**
  be a `@dataclass`: registration rejects a non-dataclass detector because its
  constructor parameters could not round-trip through `flow.yaml`.
- **Keep parameters flat scalars** (ints/floats/strings/bools). Flat params
  round-trip through `flow.yaml` via the registry; nested objects do not.
- **Emit only `sf.RISE` / `sf.FALL` / `sf.NONE`** into the `signal` column. Any
  other value is rejected when the flow runs, and a detector that never produces a
  `signal` column at all raises a clear error naming it.
- **Be causal.** Use `.over("pair")` on rolling windows so a value at bar `t`
  never depends on the future and never crosses a pair boundary. `compute()`
  sorts by `(pair, ts)` before calling `detect()`.
- **Declare `warmup`** as the number of leading bars your longest window needs, so
  the flow reserves enough history before it starts trading and the incremental
  live loop stays exactly equal to the vectorized backtest.

## Putting it in a Flow

A detector that computes its own features (like the one above) needs no
`forecasts` slot - the signal is derived directly from the data.

```python
ds = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-03-01", interval="1h")

flow = sf.Flow(
    name="sma_rise",
    detectors=[SmaAbove(n=20)],
    strategy=sf.RulesStrategy(entry=sf.Entry(size_pct=0.3), exit=sf.Exit(tp=0.03, sl=0.02)),
)

run = flow.backtest(ds, capital=10_000)
print(run.scorecard())
```

## It survives serialization and runs live unchanged

Because the detector is registered and its params are flat, the whole flow is
data:

```python
flow.save("flows/sma_rise/flow.yaml")
same = sf.Flow.load("flows/sma_rise/flow.yaml")  # detector rebuilt from the registry
```

And the signal is causal, so the vectorized backtest and the incremental
walk-forward (`flow.simulate`, the same loop that drives live) agree bar for bar:

```python
bt = flow.backtest(ds, capital=10_000)
sim = flow.simulate(ds, capital=10_000)  # recompute per bar, only past data
assert sim.final_equity == bt.final_equity  # no look-ahead
```

The shipped indicator library `signalflow-ta` provides ready-made feature and
detector building blocks; install it to reuse them instead of hand-rolling
windows.

See [Live & Walk-Forward](live-and-walk-forward.md) for the live loop and the
walk-forward refit/cache.
