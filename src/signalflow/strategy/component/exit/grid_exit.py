"""Grid-level exit rule for collective grid management."""

from dataclasses import dataclass

from signalflow.core import Order, Position, PositionType, StrategyState, sf_component
from signalflow.strategy.component.base import ExitRule


@dataclass
@sf_component(name="grid_exit")
class GridExit(ExitRule):
    """Exit rule that manages the grid as a whole.

    Unlike per-position TP/SL, GridExit operates on aggregate grid state:
    - Close ALL positions when total grid profit target is reached.
    - Close ALL positions when total grid loss limit is breached.
    - Close individual positions that have drifted too far from current price.
    """

    grid_profit_target: float | None = None
    grid_loss_limit: float | None = None
    max_distance_pct: float | None = None
    initial_capital_key: str = "initial_capital"

    def check_exits(self, positions: list[Position], prices: dict[str, float], state: StrategyState) -> list[Order]:
        open_positions = [p for p in positions if not p.is_closed]
        if not open_positions:
            return []

        # Compute aggregate grid PnL across open positions
        total_pnl = 0.0
        for pos in open_positions:
            price = prices.get(pos.pair)
            if price is None or price <= 0:
                continue
            if pos.position_type == PositionType.LONG:
                total_pnl += (price - pos.entry_price) * pos.qty
            else:
                total_pnl += (pos.entry_price - price) * pos.qty

        # Check aggregate targets
        initial_capital = state.runtime.get(self.initial_capital_key, 0.0)
        if initial_capital > 0:
            pnl_pct = total_pnl / initial_capital
        else:
            # Fallback: use sum of entry notionals as denominator
            total_notional = sum(p.entry_price * p.qty for p in open_positions)
            pnl_pct = total_pnl / total_notional if total_notional > 0 else 0.0

        # Grid profit target — close all
        if self.grid_profit_target is not None and pnl_pct >= self.grid_profit_target:
            return self._close_all(open_positions, prices, "grid_profit_target", pnl_pct)

        # Grid loss limit — close all
        if self.grid_loss_limit is not None and pnl_pct <= -self.grid_loss_limit:
            return self._close_all(open_positions, prices, "grid_loss_limit", pnl_pct)

        # Stale level — close individual positions too far from current price
        if self.max_distance_pct is not None:
            return self._close_stale(open_positions, prices)

        return []

    def _close_all(
        self,
        positions: list[Position],
        prices: dict[str, float],
        reason: str,
        pnl_pct: float,
    ) -> list[Order]:
        orders: list[Order] = []
        for pos in positions:
            price = prices.get(pos.pair)
            if price is None or price <= 0:
                continue
            side = "SELL" if pos.position_type == PositionType.LONG else "BUY"
            orders.append(
                Order(
                    pair=pos.pair,
                    side=side,
                    order_type="MARKET",
                    qty=pos.qty,
                    position_id=pos.id,
                    meta={
                        "exit_reason": reason,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "grid_pnl_pct": pnl_pct,
                    },
                )
            )
        return orders

    def _close_stale(
        self,
        positions: list[Position],
        prices: dict[str, float],
    ) -> list[Order]:
        orders: list[Order] = []
        for pos in positions:
            price = prices.get(pos.pair)
            if price is None or price <= 0:
                continue
            distance_pct = abs(price - pos.entry_price) / pos.entry_price
            if distance_pct > self.max_distance_pct:
                side = "SELL" if pos.position_type == PositionType.LONG else "BUY"
                orders.append(
                    Order(
                        pair=pos.pair,
                        side=side,
                        order_type="MARKET",
                        qty=pos.qty,
                        position_id=pos.id,
                        meta={
                            "exit_reason": "grid_stale_level",
                            "entry_price": pos.entry_price,
                            "exit_price": price,
                            "distance_pct": distance_pct,
                        },
                    )
                )
        return orders
