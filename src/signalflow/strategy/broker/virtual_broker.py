"""Virtual broker for paper trading with order/fill logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

import polars as pl
from loguru import logger

from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType
from signalflow.strategy.broker.backtest import BacktestBroker


@dataclass
@sf_component(name="virtual/realtime", override=True)
class VirtualRealtimeBroker(BacktestBroker):
    """Broker for paper trading with structured order/fill logging.

    Extends ``BacktestBroker`` with in-memory ledgers and structured
    logging of every order, fill, and trade.  Recommended for use with
    ``RealtimeRunner`` during virtual trading.

    Usage::

        broker = VirtualRealtimeBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0005),
            store=strategy_store,
        )
        runner = RealtimeRunner(broker=broker, ...)
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_BROKER

    _order_log: list[dict] = field(default_factory=list, init=False, repr=False)
    _fill_log: list[dict] = field(default_factory=list, init=False, repr=False)

    def submit_orders(
        self,
        orders: list[Order],
        prices: dict[str, float],
        ts: datetime,
    ) -> list[OrderFill]:
        """Submit orders with logging.  Delegates to executor directly."""
        if not orders:
            return []

        for order in orders:
            order_record = {
                "ts": ts,
                "order_id": order.id,
                "pair": order.pair,
                "side": order.side,
                "order_type": order.order_type,
                "qty": order.qty,
                "price": prices.get(order.pair),
                "position_id": order.position_id,
                "signal_strength": order.signal_strength,
            }
            self._order_log.append(order_record)
            logger.info(
                f"ORDER {order.side} {order.pair} qty={order.qty:.6f} "
                f"price={prices.get(order.pair, 0):.2f} "
                f"signal={order.signal_strength:.2f} "
                f"position_id={order.position_id}"
            )

        fills = self.executor.execute(orders, prices, ts)

        for fill in fills:
            fill_record = {
                "ts": fill.ts,
                "fill_id": fill.id,
                "order_id": fill.order_id,
                "pair": fill.pair,
                "side": fill.side,
                "price": fill.price,
                "qty": fill.qty,
                "fee": fill.fee,
                "position_id": fill.position_id,
            }
            self._fill_log.append(fill_record)
            logger.info(
                f"FILL {fill.side} {fill.pair} qty={fill.qty:.6f} "
                f"price={fill.price:.2f} fee={fill.fee:.4f} "
                f"fill_id={fill.id[:8]}"
            )

        return fills

    def process_fills(self, fills: list[OrderFill], orders: list[Order], state: StrategyState) -> list[Trade]:
        """Process fills with logging.  Delegates to ``BacktestBroker``."""
        trades = super().process_fills(fills, orders, state)

        for trade in trades:
            trade_type = trade.meta.get("type", "unknown")
            logger.info(
                f"TRADE [{trade_type}] {trade.side} {trade.pair} "
                f"qty={trade.qty:.6f} price={trade.price:.2f} "
                f"fee={trade.fee:.4f} position={trade.position_id}"
            )

        return trades

    @property
    def order_log(self) -> list[dict]:
        """All orders submitted during this session."""
        return self._order_log

    @property
    def fill_log(self) -> list[dict]:
        """All fills received during this session."""
        return self._fill_log

    def order_log_df(self) -> pl.DataFrame:
        """Orders as a Polars DataFrame for analysis."""
        if not self._order_log:
            return pl.DataFrame()
        return pl.DataFrame(self._order_log)

    def fill_log_df(self) -> pl.DataFrame:
        """Fills as a Polars DataFrame for analysis."""
        if not self._fill_log:
            return pl.DataFrame()
        return pl.DataFrame(self._fill_log)
