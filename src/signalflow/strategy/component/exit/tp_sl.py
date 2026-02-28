from dataclasses import dataclass
from typing import Literal, cast

from loguru import logger

from signalflow.core import Order, Position, PositionType, StrategyState, exit
from signalflow.strategy.component.base import ExitRule


@dataclass
@exit("tp_sl")
class TakeProfitStopLossExit(ExitRule):
    """
    Exit rule based on take-profit and stop-loss levels.

    Can use fixed percentages or dynamic levels from position meta.
    """

    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    use_position_levels: bool = False

    def check_exits(self, positions: list[Position], prices: dict[str, float], state: StrategyState) -> list[Order]:
        orders: list[Order] = []

        for pos in positions:
            if pos.is_closed:
                continue

            price = prices.get(pos.pair)
            if price is None or price <= 0:
                continue

            if self.use_position_levels:
                tp_price = pos.meta.get("take_profit_price")
                sl_price = pos.meta.get("stop_loss_price")
            else:
                if pos.position_type == PositionType.LONG:
                    tp_price = pos.entry_price * (1 + self.take_profit_pct)
                    sl_price = pos.entry_price * (1 - self.stop_loss_pct)
                else:
                    tp_price = pos.entry_price * (1 - self.take_profit_pct)
                    sl_price = pos.entry_price * (1 + self.stop_loss_pct)

            should_exit = False
            exit_reason = ""

            if pos.position_type == PositionType.LONG:
                if tp_price and price >= tp_price:
                    should_exit = True
                    exit_reason = "take_profit"
                elif sl_price and price <= sl_price:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:
                if tp_price and price <= tp_price:
                    should_exit = True
                    exit_reason = "take_profit"
                elif sl_price and price >= sl_price:
                    should_exit = True
                    exit_reason = "stop_loss"

            if should_exit:
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                if pos.position_type != PositionType.LONG:
                    pnl_pct = -pnl_pct
                logger.debug(
                    "{} exit {}: entry={:.2f} → {:.2f} ({:+.2f}%)",
                    exit_reason, pos.pair, pos.entry_price, price, pnl_pct,
                )
                side = cast(Literal["BUY", "SELL"], "SELL" if pos.position_type == PositionType.LONG else "BUY")
                order = Order(
                    pair=pos.pair,
                    side=side,
                    order_type="MARKET",
                    qty=pos.qty,
                    position_id=pos.id,
                    meta={
                        "exit_reason": exit_reason,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                    },
                )
                orders.append(order)

        return orders
