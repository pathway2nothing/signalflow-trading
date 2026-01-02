# IMPORTANT

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from signalflow.core.containers import Position, Trade
from signalflow.strategy.types import StrategyContext, NewPositionOrder
from signalflow.strategy.state import StrategyState
from signalflow.strategy.store.base import StrategyStore
from signalflow.strategy.executor.base import OrderExecutor
from signalflow.strategy.component.base import StrategyEntryRule, StrategyExitRule, StrategyMetric


@dataclass
class StrategyEngine:
    store: StrategyStore
    signal_source: SignalSource
    execution_model: OrderExecutor

    entry: StrategyEntryRule
    exits: list[StrategyExitRule]
    metrics: list[StrategyMetric]

    def bootstrap(self, state: StrategyState) -> StrategyState:
        state.portfolio = self.store.load_portfolio(state.strategy_id)
        open_positions = self.store.load_open_positions(state.strategy_id)
        state.portfolio.positions = {p.id: p for p in open_positions}
        state.last_event_id = self.store.kv_get(state.strategy_id, "last_event_id")
        return state

    def step(self, *, state: StrategyState, ts: datetime, event_id: str | None, signals, prices: dict[str, float]) -> StrategyState:
        if event_id is not None:
            if state.last_event_id == event_id:
                return state

        ctx = StrategyContext(strategy_id=state.strategy_id, ts=ts, prices=prices, runtime=state.runtime)

        fills = self.execution_model.sync_fills(state=state, ts=ts)
        if fills:
            self._apply_fills(state, fills)
            self.store.insert_trades(state.strategy_id, fills)
            self.store.upsert_positions(state.strategy_id, list(state.portfolio.positions.values()))

        for p in state.portfolio.open_positions():
            px = prices.get(p.pair)
            if px is not None:
                p.mark(ts=ts, price=float(px))
        self.store.upsert_positions(state.strategy_id, list(state.portfolio.positions.values()))

        metrics_row: dict[str, float] = {}
        for m in self.metrics:
            metrics_row.update(m.compute(state=state, context=ctx))
        self.store.insert_metrics(state.strategy_id, ts, metrics_row)

        to_close: list[tuple[Position, str]] = []
        for p in state.portfolio.open_positions():
            for ex in self.exits:
                ok, reason = ex.should_exit(position=p, state=state, context=ctx)
                if ok:
                    to_close.append((p, reason))
                    break

        if to_close:
            close_ids = [p.id for p, _ in to_close]
            self.store.close_positions(state.strategy_id, close_ids, ts, reason=";".join({r for _, r in to_close}))
            open_positions = self.store.load_open_positions(state.strategy_id)
            state.portfolio.positions = {p.id: p for p in open_positions}

        orders: list[NewPositionOrder] = self.entry.build_orders(signals=signals, state=state, context=ctx)
        if orders:
            new_pos, new_trades = self.executor.open_positions(state=state, orders=orders, ts=ts, prices=prices)
            for p in new_pos:
                state.portfolio.positions[p.id] = p
            if new_trades:
                self.store.insert_trades(state.strategy_id, new_trades)
            self.store.upsert_positions(state.strategy_id, new_pos)

        self.store.save_portfolio(state.strategy_id, state.portfolio)

        state.last_ts = ts
        if event_id is not None:
            state.last_event_id = event_id
            self.store.kv_set(state.strategy_id, "last_event_id", event_id)

        return state

    def _apply_fills(self, state: StrategyState, fills: list[Trade]) -> None:
        for t in fills:
            if t.position_id is None:
                continue
            p = state.portfolio.positions.get(t.position_id)
            if p is None:
                continue
            p.apply_trade(t)
            state.portfolio.positions[p.id] = p
