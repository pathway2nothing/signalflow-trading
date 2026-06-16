# Live & Walk-Forward

One decision core drives backtest, paper, and live. A flow that backtests
correctly runs live unchanged - the only difference is where the bars come from.

## Backtest == walk-forward (no look-ahead)

`flow.backtest(data)` precomputes signals over a finished Dataset (vectorized,
fast). `flow.simulate(data)` replays the **same loop used live**: it feeds bars
one at a time, recomputing signals over only the data seen so far. If the signal
is causal, the two agree exactly:

```python
bt = flow.backtest(ds, capital=10_000)
sim = flow.simulate(ds, capital=10_000)        # incremental; the live decision core
assert sim.final_equity == bt.final_equity
assert len(sim.fills) == len(bt.fills)
```

A mismatch means look-ahead leaked into a feature or detector. `simulate` is much
slower than `backtest` (it recomputes per bar), so use it to validate a flow, not
for routine research. `simulate(warmup=N)` reserves a leading window that fills
the buffer without trading - a train/test split for walk-forward.

## Going live

```python
feed = sf.PollingFeed(sf.BinanceSource(), pairs=["BTCUSDT"], interval="1m")

flow.live(feed, capital=10_000)                 # live data, sim fills (paper)
flow.live(feed, capital=10_000, armed=True,     # real orders
          broker=sf.BinanceBroker(api_key=..., api_secret=...),
          state_path="book.json")               # book persists across restarts
```

`PollingFeed` warms a rolling buffer from history, then waits for each freshly
**closed** bar (never the still-forming one) and yields it. The loop records the
gap between the bar's close and order execution; a breach of the latency budget
is logged.

## Rolling refit on a trailing window

When a flow's `ForecastModel` uses a WoE encoder, the walk-forward refits the
whole stack on a schedule set by the encoder's `refit` (step) and `window`
(trailing train span):

```python
from signalflow.transform.encode import WoE

model = sf.ForecastModel(
    target=sf.FixedHorizon(bars=12),
    features=sf.FeaturePipe(sf.SMA(10), sf.SMA(20)),
    encode=WoE(refit="1d", window="365d"),      # refit daily on a trailing year
)
model.fit(ds)
```

Each refit fits fresh bin edges + WoE/IV tables on its window. The binning can
shift from one refit to the next - that is expected; every refit is recorded.

## Inspecting and caching the refit history

The per-refit statistics (bin edges + WoE table + IV, tagged with the target and
fold window) are kept on the fitted model and are portable:

```python
hist = model.woe_history()       # [{test_start, train_start, train_end, target, state}, ...]
model.dump_woe_history("woe_history.json")
```

A single feature can have several WoE variants - different targets, different
binnings - and each fold's table is stored separately, so nothing collides.

For long histories, pass an `ArtifactCache` so a re-fit recomputes only **new or
changed** folds instead of the whole timeline:

```python
from signalflow.experiment import ArtifactCache

cache = ArtifactCache("cache/folds")
model.fit(ds, cache=cache)        # first run computes + stores each fold
model.fit(ds, cache=cache)        # re-run loads unchanged folds, computes only new ones
```

The cache key folds in the feature/encoder/target config, the code fingerprint,
the dataset identity, and the fold's window bounds - so editing a feature or
changing the data invalidates the affected folds automatically.

A fitted `FeaturePipe` or any transform tree is also serializable on its own:

```python
pipe = sf.FeaturePipe(sf.SMA(10), sf.SMA(20))
pipe.save("pipe.yaml")
same = sf.FeaturePipe.load("pipe.yaml")
```
