"""Virtual broker for paper trading with order/fill logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
from loguru import logger

from signalflow.core.containers.order import Order, OrderFill
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType
from signalflow.strategy.broker.backtest import BacktestBroker

if TYPE_CHECKING:
    from signalflow.strategy.risk.manager import RiskManager


@dataclass
@sf_component(name="virtual/realtime", override=True)
class VirtualRealtimeBroker(BacktestBroker):
    """Broker for paper trading with structured order/fill logging.

    Extends ``BacktestBroker`` with:
    - In-memory order/fill/trade ledgers
    - Structured logging of every order, fill, and trade
    - Optional :class:`RiskManager` integration for pre-trade checks
    - Equity curve tracking for post-session analysis

    Usage::

        from signalflow.strategy.risk import RiskManager, MaxLeverageLimit

        broker = VirtualRealtimeBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0005),
            store=strategy_store,
            risk_manager=RiskManager(limits=[MaxLeverageLimit(max_leverage=3.0)]),
        )
        runner = RealtimeRunner(broker=broker, ...)
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_BROKER

    risk_manager: RiskManager | None = None

    _order_log: list[dict] = field(default_factory=list, init=False, repr=False)
    _fill_log: list[dict] = field(default_factory=list, init=False, repr=False)
    _equity_curve: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def submit_orders(
        self,
        orders: list[Order],
        prices: dict[str, float],
        ts: datetime,
    ) -> list[OrderFill]:
        """Submit orders with risk checking and logging."""
        if not orders:
            return []

        # Risk manager gate
        if self.risk_manager is not None:
            # We need StrategyState for risk checks; build a minimal one from
            # the broker's perspective using the last known state.  The caller
            # (runner) should have already marked positions, so
            # ``self._last_state`` is up-to-date.
            state = getattr(self, "_last_state", None)
            if state is not None:
                result = self.risk_manager.check(orders, state, prices, ts)
                if not result.allowed and not result.passed_orders:
                    for name, reason in result.violations:
                        logger.warning(f"RISK REJECT [{name}] {reason}")
                    return []
                orders = result.passed_orders

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

    def mark_positions(self, state: StrategyState, prices: dict[str, float], ts: datetime) -> None:
        """Mark positions and record equity snapshot."""
        super().mark_positions(state, prices, ts)
        # Keep reference for risk manager
        self._last_state = state
        # Track equity curve
        self._equity_curve.append(
            {
                "ts": ts,
                "equity": state.portfolio.equity(prices=prices),
                "cash": state.portfolio.cash,
                "n_positions": len(state.portfolio.open_positions()),
                "gross_exposure": state.portfolio.gross_exposure(prices=prices),
                "net_exposure": state.portfolio.net_exposure(prices=prices),
            }
        )

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

    def equity_curve_df(self) -> pl.DataFrame:
        """Equity curve as a Polars DataFrame."""
        if not self._equity_curve:
            return pl.DataFrame()
        return pl.DataFrame(self._equity_curve)

    @property
    def equity_curve(self) -> list[dict[str, Any]]:
        """Raw equity curve records."""
        return self._equity_curve

    @property
    def last_state(self) -> StrategyState | None:
        """Last known strategy state (from most recent mark_positions)."""
        return getattr(self, "_last_state", None)

    def get_account_summary(self, initial_capital: float = 10_000.0) -> dict[str, Any]:
        """Get current account summary.

        Args:
            initial_capital: Starting capital for PnL calculation.

        Returns:
            Dict with balance, equity, open_positions, total_pnl, return_pct.
        """
        state = self.last_state
        eq = self._equity_curve

        if state and eq:
            last = eq[-1]
            equity = last.get("equity", initial_capital)
            cash = last.get("cash", initial_capital)
        elif state:
            cash = state.portfolio.cash if hasattr(state.portfolio, "cash") else initial_capital
            equity = cash
        else:
            cash = initial_capital
            equity = initial_capital

        n_open = 0
        if state and hasattr(state.portfolio, "open_positions"):
            n_open = len(state.portfolio.open_positions())

        return {
            "initial_capital": initial_capital,
            "balance": round(cash, 2),
            "equity": round(equity, 2),
            "open_positions": n_open,
            "total_pnl": round(equity - initial_capital, 2),
            "return_pct": round((equity - initial_capital) / initial_capital, 4) if initial_capital > 0 else 0,
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Get open positions in a serializable format.

        Returns:
            List of position dicts with pair, side, entry_price, size, pnl.
        """
        state = self.last_state
        if not state or not hasattr(state.portfolio, "open_positions"):
            return []

        result: list[dict[str, Any]] = []
        for pos in state.portfolio.open_positions():
            entry_price = getattr(pos, "entry_price", 0)
            last_price = getattr(pos, "last_price", entry_price)
            qty = getattr(pos, "qty", 0)

            from signalflow.core.containers.position import PositionType

            is_long = getattr(pos, "position_type", PositionType.LONG) == PositionType.LONG
            pnl = (last_price - entry_price) * qty if is_long else (entry_price - last_price) * qty
            notional = entry_price * qty
            pnl_pct = pnl / notional if notional > 0 else 0

            result.append({
                "id": getattr(pos, "id", ""),
                "pair": getattr(pos, "pair", ""),
                "side": "long" if is_long else "short",
                "entry_price": round(entry_price, 6),
                "size": round(qty, 8),
                "entry_time": pos.entry_time.isoformat() if getattr(pos, "entry_time", None) else None,
                "pnl": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 4),
            })
        return result
