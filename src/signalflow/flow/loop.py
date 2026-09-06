"""The decision loop - one loop for backtest/paper/live."""

import time
from typing import Any

import polars as pl
from loguru import logger

from signalflow._logging import frame_summary, step
from signalflow.engine.engine import Engine
from signalflow.engine.types import Order
from signalflow.enums import FALL, NONE, RISE, SIGNAL_COL, IntentKind, OrderType, RunMode
from signalflow.errors import PipelineError
from signalflow.flow.bundle import MIN_OOS_COVERAGE
from signalflow.strategy.observation import Observation
from signalflow.transform.base import ensure_sorted

EMPTY_SIGNALS_SCHEMA: dict[str, Any] = {
    "pair": pl.Utf8,
    "ts": pl.Datetime("ms"),
    "signal": pl.Utf8,
    "p_success": pl.Float64,
}

def enriched_signals(flow: Any, data: Any, oos: bool = False, log: bool = True) -> pl.DataFrame:
    """Precompute forecast columns, run detectors, and (event-gated) validator scores.

    When ``oos`` is true, every forecast slot and the validator use leak-free
    out-of-fold predictions; rows outside the model's OOS coverage are null, so
    detectors do not fire there. Leave it false for the production in-sample path.
    ``log=False`` silences the per-slot/per-detector DEBUG lines (the live loop
    calls this every bar and reports progress itself).
    """
    enriched = data
    for slot, model in flow.forecasts.items():
        out = getattr(model, "output", "p_rise")
        t0 = time.perf_counter()
        pred = model.predict_oos(data) if oos else model.predict(data)
        if log:
            logger.debug(
                f"forecast slot {slot!r}: {'oos' if oos else 'in-sample'} rows={pred.height:,} "
                f"non-null={pred.height - pred.get_column(out).null_count():,} ({time.perf_counter() - t0:.2f}s)"
            )
        pred = pred.rename({out: f"{slot}/{out}"})
        enriched = enriched.with_forecasts(pred)

    parts = []
    for det in flow.detectors:
        t0 = time.perf_counter()
        try:
            computed = det.compute(enriched.frame)
        except Exception as e:
            raise PipelineError(f"detector {det.name!r} failed during compute: {e}") from e
        if SIGNAL_COL not in computed.columns:
            raise PipelineError(
                f"detector {det.name!r} did not produce the {SIGNAL_COL!r} column; produced columns: {computed.columns}"
            )
        emitted = set(computed.get_column(SIGNAL_COL).drop_nulls().unique().to_list())
        invalid = emitted - {RISE, FALL, NONE}
        if invalid:
            raise PipelineError(
                f"detector {det.name!r} emitted invalid signal values {sorted(invalid)}; "
                f"expected only {sorted({RISE, FALL, NONE})}"
            )
        s = (
            computed.filter(pl.col(SIGNAL_COL) != NONE)
            .select(["pair", "ts", SIGNAL_COL])
            .with_columns(pl.lit(det.name).alias("detector"))
        )
        if log:
            sig = s.get_column(SIGNAL_COL)
            logger.debug(
                f"detector {det.name!r}: signals={s.height:,} (rise={(sig == RISE).sum():,}, "
                f"fall={(sig == FALL).sum():,}) over rows={computed.height:,} ({time.perf_counter() - t0:.2f}s)"
            )
        parts.append(s)
    signals = pl.concat(parts) if parts else pl.DataFrame(schema={**EMPTY_SIGNALS_SCHEMA, "detector": pl.Utf8})

    if flow.validator is not None and signals.height > 0:
        vcol = getattr(flow.validator, "output", "p_success")
        t0 = time.perf_counter()
        vpred = flow.validator.predict_oos(data) if oos else flow.validator.predict(data)
        vp = vpred.select(["pair", "ts", vcol]).rename({vcol: "p_success"})
        signals = signals.join(vp, on=["pair", "ts"], how="left")
        if log:
            scored = signals.height - signals.get_column("p_success").null_count()
            logger.debug(f"validator: scored {scored:,}/{signals.height:,} signals ({time.perf_counter() - t0:.2f}s)")
    return signals


def orders_from_intents(intents: Any, prices: dict[str, float], ts: Any) -> list[Order]:
    orders = []
    for it in intents:
        price = prices.get(it.pair)
        if price is None:
            continue
        otype = OrderType.LIMIT if it.limit_price is not None else OrderType.MARKET
        ref = it.limit_price if it.limit_price is not None else price
        if it.kind == IntentKind.OPEN:
            qty = (it.notional or 0.0) / ref
            if qty > 0:
                orders.append(
                    Order(it.pair, it.side, qty, type=otype, limit_price=it.limit_price, ts=ts, reason=it.reason)
                )
        elif it.qty and it.qty > 0:
            orders.append(
                Order(it.pair, it.side, it.qty, type=otype, limit_price=it.limit_price, ts=ts, reason=it.reason)
            )
    return orders



def _oos_coverage(flow: Any, data: Any) -> "float | None":
    if not flow.forecasts:
        return None
    fractions = []
    for model in flow.forecasts.values():
        pred = model.predict_oos(data)
        col = getattr(model, "output", "p_rise")
        fractions.append(1.0 - pred.get_column(col).null_count() / max(pred.height, 1))
    return float(min(fractions))


def run_event_loop(
    flow: Any,
    data: Any,
    capital: Any,
    target: Any,
    broker: Any,
    mode: RunMode,
    mandate: dict | None = None,
    oos: bool = False,
) -> Any:
    from signalflow.flow.run import Run

    target = target or data.quote
    t_run = time.perf_counter()
    logger.debug(
        f"Flow.{mode.value}({flow.name!r}): capital={capital} target={target} oos={oos} "
        f"strategy={getattr(flow.strategy, 'name', type(flow.strategy).__name__)} data: {frame_summary(data.frame)}"
    )
    engine = Engine(capital, target=target, quote=data.quote)
    with step(f"Flow.{mode.value}: signals", forecasts=len(flow.forecasts), detectors=len(flow.detectors)) as log:
        signals = enriched_signals(flow, data, oos=oos)
        log["signals"] = f"{signals.height:,}"
    if data.frame.height == 0:
        logger.warning(f"backtest of {flow.name!r}: dataset is empty (0 bars)")
    by_ts: dict = {}
    if signals.height:
        for key, df in signals.group_by("ts"):
            by_ts[key[0] if isinstance(key, tuple) else key] = df

    eq_ts, eq_val = [], []
    peak = float("-inf")
    started = False
    n_bars = n_intents = n_orders = n_fills = 0
    t_loop = time.perf_counter()
    fill_mode = getattr(broker, "fill", "close")
    pending: list = []
    for bar in data.iter_bars():
        n_bars += 1
        if pending:
            fills = broker.execute(pending, bar, at="open")
            pending = []
            n_fills += len(fills)
            engine.apply(fills)
        snap = engine.snapshot(bar.ts, bar.prices)
        if not started:
            eq_ts.append(bar.ts)
            eq_val.append(snap.equity)
            started = True
        peak = max(peak, snap.equity)
        sig_frame = by_ts.get(bar.ts)
        if sig_frame is None:
            sig_frame = pl.DataFrame(schema=EMPTY_SIGNALS_SCHEMA)
        obs = Observation(bar.ts, sig_frame, snap, mandate or {})
        intents = flow.strategy.decide(obs)
        n_intents += len(intents)
        intents = flow.risk.clip(intents, snap, peak)
        orders = orders_from_intents(intents, bar.prices, bar.ts)
        n_orders += len(orders)
        if fill_mode == "next_open":
            pending = orders
            fills = []
        else:
            fills = broker.execute(orders, bar)
        n_fills += len(fills)
        engine.apply(fills)
        eq_ts.append(bar.ts)
        eq_val.append(engine.equity(bar.prices))
    logger.debug(
        f"Flow.{mode.value}: loop bars={n_bars:,} intents={n_intents:,} orders={n_orders:,} fills={n_fills:,} "
        f"({time.perf_counter() - t_loop:.2f}s)"
    )

    curve = pl.DataFrame({"ts": eq_ts, "equity": eq_val})
    coverage = _oos_coverage(flow, data) if oos else None
    if coverage is not None and coverage < 1.0:
        logger.warning(
            f"backtest of {flow.name!r}: only {coverage:.1%} of requested rows are covered by cached OOS predictions"
        )
    # A rule-only flow is in-sample too (its thresholds were tuned on some span): the caller asserts
    # out-of-sample evidence with oos=True; a model flow additionally needs enough OOS coverage.
    promotable = oos and (coverage is None or coverage >= MIN_OOS_COVERAGE)
    run = Run(
        flow.name, mode.value, curve, engine.event_log, target, promotable=promotable, oos=oos, oos_coverage=coverage
    )
    logger.info(
        f"Flow.{mode.value}({flow.name!r}): bars={n_bars:,} signals={signals.height:,} fills={n_fills:,} "
        f"equity {run.initial_equity:,.2f} -> {run.final_equity:,.2f} ({run.total_return:+.2%}) "
        f"max_dd={run.max_drawdown:.2%} promotable={promotable} ({time.perf_counter() - t_run:.2f}s)"
    )
    return run


def run_quicktest(flow: Any, data: Any, capital: Any, target: Any, horizon: int = 24, fee: float = 0.001) -> Any:
    """Vectorized triage: forward return per RISE signal. NOT promotable."""
    from signalflow.flow.run import Run

    target = target or data.quote
    enriched = data
    for slot, model in flow.forecasts.items():
        out = getattr(model, "output", "p_rise")
        enriched = enriched.with_forecasts(model.predict(data).rename({out: f"{slot}/{out}"}))

    frame = ensure_sorted(enriched.frame).with_columns(
        (pl.col("close").shift(-horizon).over("pair") / pl.col("close") - 1.0).alias("_fwd")
    )
    sig_parts = [
        d.compute(frame).filter(pl.col("signal") == RISE).select(["pair", "ts", "_fwd"]) for d in flow.detectors
    ]
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


__all__ = [
    "EMPTY_SIGNALS_SCHEMA",
    "enriched_signals",
    "orders_from_intents",
    "run_event_loop",
    "run_quicktest",
]
