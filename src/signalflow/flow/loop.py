"""The decision loop - one loop for backtest/paper/live."""


import polars as pl

from signalflow.engine.engine import Engine
from signalflow.engine.types import Order
from signalflow.enums import NONE, RISE, IntentKind, RunMode
from signalflow.strategy.observation import Observation

_EMPTY_SIGNALS_SCHEMA = {"pair": pl.Utf8, "ts": pl.Datetime("ms"), "signal": pl.Utf8, "p_success": pl.Float64}


def enriched_signals(flow, data) -> pl.DataFrame:
    """Precompute forecast columns, run detectors, and (event-gated) validator scores."""
    enriched = data
    for slot, model in flow.forecasts.items():
        out = getattr(model, "output", "p_rise")
        pred = model.predict(data).rename({out: f"{slot}/{out}"})
        enriched = enriched.with_forecasts(pred)

    parts = []
    for det in flow.detectors:
        s = (
            det.compute(enriched.frame)
            .filter(pl.col("signal") != NONE)
            .select(["pair", "ts", "signal"])
            .with_columns(pl.lit(det.name).alias("detector"))
        )
        parts.append(s)
    signals = pl.concat(parts) if parts else pl.DataFrame(schema={**_EMPTY_SIGNALS_SCHEMA, "detector": pl.Utf8})

    if flow.validator is not None and signals.height > 0:
        vcol = getattr(flow.validator, "output", "p_success")
        vp = flow.validator.predict(data).select(["pair", "ts", vcol]).rename({vcol: "p_success"})
        signals = signals.join(vp, on=["pair", "ts"], how="left")
    return signals


def _orders(intents, prices, ts):
    orders = []
    for it in intents:
        price = prices.get(it.pair)
        if price is None:
            continue
        if it.kind == IntentKind.OPEN:
            qty = (it.notional or 0.0) / price
            if qty > 0:
                orders.append(Order(it.pair, it.side, qty, ts=ts, reason=it.reason))
        elif it.qty and it.qty > 0:
            orders.append(Order(it.pair, it.side, it.qty, ts=ts, reason=it.reason))
    return orders


def run_event_loop(flow, data, capital, target, broker, mode: RunMode, mandate: dict | None = None):
    from signalflow.flow.run import Run

    target = target or data.quote
    engine = Engine(capital, target=target, quote=data.quote)
    signals = enriched_signals(flow, data)
    by_ts: dict = {}
    if signals.height:
        for key, df in signals.group_by("ts"):
            by_ts[key[0] if isinstance(key, tuple) else key] = df

    eq_ts, eq_val = [], []
    peak = float("-inf")
    for bar in data.iter_bars():
        snap = engine.snapshot(bar.ts, bar.prices)
        peak = max(peak, snap.equity)
        eq_ts.append(bar.ts)
        eq_val.append(snap.equity)
        sig_frame = by_ts.get(bar.ts)
        if sig_frame is None:
            sig_frame = pl.DataFrame(schema=_EMPTY_SIGNALS_SCHEMA)
        obs = Observation(bar.ts, sig_frame, snap, mandate or {})
        intents = flow.strategy.decide(obs)
        intents = flow.risk.clip(intents, snap, peak)
        fills = broker.execute(_orders(intents, bar.prices, bar.ts), bar)
        engine.apply(fills)

    curve = pl.DataFrame({"ts": eq_ts, "equity": eq_val})
    return Run(flow.name, mode.value, curve, engine.event_log, target, promotable=True)


def run_quicktest(flow, data, capital, target, horizon: int = 24, fee: float = 0.001):
    """Vectorized triage: forward return per RISE signal. NOT promotable."""
    from signalflow.flow.run import Run

    target = target or data.quote
    enriched = data
    for slot, model in flow.forecasts.items():
        out = getattr(model, "output", "p_rise")
        enriched = enriched.with_forecasts(model.predict(data).rename({out: f"{slot}/{out}"}))

    frame = enriched.frame.sort(["pair", "ts"]).with_columns(
        (pl.col("close").shift(-horizon).over("pair") / pl.col("close") - 1.0).alias("_fwd")
    )
    sig_parts = [d.compute(frame).filter(pl.col("signal") == RISE).select(["pair", "ts", "_fwd"]) for d in flow.detectors]
    rises = pl.concat(sig_parts) if sig_parts else frame.head(0).select(["pair", "ts", "_fwd"])
    rises = rises.drop_nulls("_fwd").sort("ts")

    size_pct = getattr(getattr(flow.strategy, "entry", None), "size_pct", 0.1)
    equity = float(capital if isinstance(capital, (int, float)) else sum(capital.values()))
    eq_ts, eq_val = [], []
    for row in rises.iter_rows(named=True):
        equity *= 1.0 + size_pct * (row["_fwd"] - 2 * fee)
        eq_ts.append(row["ts"])
        eq_val.append(equity)
    if not eq_ts:
        eq_ts, eq_val = [frame.get_column("ts").min()], [equity]
    curve = pl.DataFrame({"ts": eq_ts, "equity": eq_val})
    return Run(flow.name, RunMode.QUICKTEST.value, curve, [], target, promotable=False)
