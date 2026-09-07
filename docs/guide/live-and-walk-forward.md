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

## Stopping a running loop

`Risk(kill_switch_path="kill")` makes that file the switch: while it exists every new
entry is dropped (closes still pass), and the check runs on every bar, so creating or
deleting the file from outside engages or releases a running `live`/`simulate`. A
drawdown breach trips the switch the same way (and writes the file); `risk.reset()`
releases it explicitly. The trip is saved with the book, so a loop resumed from
`state_path` starts tripped when it was tripped. Armed runs raise
`KillSwitchTripped` instead of silently dropping orders.

## Bring your own loop

A runner that already owns its feed, portfolio store and executor does not need
`flow.live`: the loop's body is a public, stateless call. `flow.buffer()` keeps
the trailing `required_warmup + 1` bars (appends are cheap, trims are zero-copy
slices), and `flow.decide(history, snapshot, ts)` returns what the flow would do
on that bar - the signals it saw, the intents after the risk layer, and the
orders to send - without touching any state:

```python
flow = sf.Flow.load("flows/my-flow/flow.yaml")
flow.check_warmup()                                   # refuse under-declared components up front

buf = flow.buffer()                                   # window = required_warmup + 1
buf.push(my_feed.history(pairs, bars=flow.required_warmup))

peak = my_store.peak_equity()
for bar in my_feed.closed_bars():                     # one frame per closed bar, all pairs
    buf.push(bar.frame)
    snap = sf.PortfolioSnapshot(                      # your books, in the type the strategy reads
        ts=bar.ts, target="USDT", balances=my_store.balances(),
        positions=my_store.positions(), equity=my_store.equity(bar.prices), prices=bar.prices,
    )
    d = flow.decide(buf, snap, bar.ts, peak=peak, mandate={"max_notional": 5_000})
    peak = max(peak, snap.equity)
    my_store.save_signals(d.signals)                  # pair, ts, signal at this bar
    my_executor.send(d.orders)                        # already risk-clipped and sized
```

`simulate` and `live` are built from the same two pieces, so an external loop
agrees with them bar for bar. For paper trading against a live ticker instead of
the bar's close, `SimBroker.execute(orders, bar, prices=ticker)` fills at the
given prices with the broker's slippage and fees.

## Calendar walk-forward

`sf.walk_forward` turns a declarative model template into per-fold models trained
on trailing calendar windows and evaluated on the next step. Each fold predicts
with a **hot warmup window** - the trailing `warmup` bars are prepended to the test
slice so features start valid, exactly as production would see them.

```python
model = sf.ForecastModel(
    target=sf.FixedHorizon(bars=12),
    features=sf.FeaturePipeline(sf.SMA(20), sf.SMA(50)),   # raw features: no stateful step
)
result = sf.walk_forward(model, ds, train="90d", step="30d")

scores = result.evaluate(lambda oos: float(oos.get_column("p_rise").mean()))
print(scores)                 # one row per fold with its window bounds and score
merged = result.oos()         # all folds' out-of-sample rows, deduped by (pair, ts)
```

Pass `save_to="mlflow://models/exp_{fold}"` to persist each fold's fitted model;
the `{fold}` placeholder is filled with the fold index.

## Rolling refit on a trailing window

`ForecastModel.fit` refits the whole stack (the pipeline's stateful tail - `WoE`,
`IVSelector`, `Scaler` - and the estimator) per fold; the fold layout is the model's `cv` scheme - `sf.Rolling(step, window)`
(default `Rolling("7d", "365d")`) or `sf.KFold(n)`:

```python
model = sf.ForecastModel(
    target=sf.FixedHorizon(bars=12),
    features=sf.FeaturePipeline(sf.SMA(10), sf.SMA(20), sf.WoE(), sf.IVSelector()),
    cv=sf.Rolling(step="1d", window="365d"),   # refit daily on a trailing year
)
model.fit(ds)
```

Each refit fits fresh bin edges + WoE/IV tables on its window. The binning can
shift from one refit to the next - that is expected; every refit is recorded.

## Comparing models and folds

`sf.scorecard_table` gives one row per model - or per walk-forward fold - with
`n_test`, `prevalence`, the firing `threshold` and `f1 / precision / recall / pr_auc /
roc_auc / brier`, so candidates are compared on one frame instead of hand-rolled loops:

```python
result = sf.walk_forward(model, ds, train="90d", step="30d")
table = sf.scorecard_table(result, ds, operating="train_q0.9")   # threshold calibrated on each fold's train window
sf.scorecard_means(table, by="target")                          # averaged per target, with row counts

sf.scorecard_table({"h12": m12, "h24": m24}, ds, operating=0.6)  # fitted models on their OOS predictions
```

The operating point is a fixed threshold, `"q<quantile>"` of the evaluated scores,
or `"train_q<quantile>"` - the quantile of the scores over the training window,
which is what a live threshold calibrated on the past would have been. Each fold
also carries a `tag` (`YYYYMM` of its test start); `walk_forward(save_to="..._{tag}")`
names saved fold models by it.

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

A fitted `FeaturePipeline` or any transform tree is also serializable on its own:

```python
pipe = sf.FeaturePipeline(sf.SMA(10), sf.SMA(20))
pipe.save("pipe.yaml")
same = sf.FeaturePipeline.load("pipe.yaml")
```
