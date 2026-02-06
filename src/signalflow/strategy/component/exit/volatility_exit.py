from dataclasses import dataclass

from signalflow.core import Order, Position, PositionType, StrategyState, sf_component
from signalflow.strategy.component.base import ExitRule


@dataclass
@sf_component(name="volatility_exit")
class VolatilityExit(ExitRule):
    """Dynamic TP/SL based on recent volatility (ATR).

    TP and SL levels are calculated as multiples of ATR from entry price.
    Levels can be fixed at entry time or recalculated each bar.

    Args:
        tp_atr_mult: TP = entry + N*ATR (LONG) or entry - N*ATR (SHORT).
        sl_atr_mult: SL = entry - N*ATR (LONG) or entry + N*ATR (SHORT).
        use_entry_atr: If True, use ATR at entry time (fixed levels).
                       If False, recalculate levels each bar.

    Example:
        >>> rule = VolatilityExit(tp_atr_mult=3.0, sl_atr_mult=1.5)
        >>> # TP at 3x ATR profit, SL at 1.5x ATR loss
    """

    tp_atr_mult: float = 3.0
    sl_atr_mult: float = 1.5
    use_entry_atr: bool = True

    # Internal meta keys
    _tp_price_key: str = "_vol_tp_price"
    _sl_price_key: str = "_vol_sl_price"
    _atr_used_key: str = "_vol_atr_used"

    def check_exits(
        self, positions: list[Position], prices: dict[str, float], state: StrategyState
    ) -> list[Order]:
        orders: list[Order] = []

        for pos in positions:
            if pos.is_closed:
                continue

            price = prices.get(pos.pair)
            if price is None or price <= 0:
                continue

            # Get or calculate TP/SL levels
            tp_price, sl_price, atr_used = self._get_levels(pos, state)

            if tp_price is None or sl_price is None:
                continue

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            if pos.position_type == PositionType.LONG:
                if price >= tp_price:
                    should_exit = True
                    exit_reason = "volatility_tp"
                elif price <= sl_price:
                    should_exit = True
                    exit_reason = "volatility_sl"
            else:
                if price <= tp_price:
                    should_exit = True
                    exit_reason = "volatility_tp"
                elif price >= sl_price:
                    should_exit = True
                    exit_reason = "volatility_sl"

            if should_exit:
                side = "SELL" if pos.position_type == PositionType.LONG else "BUY"
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
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "atr_used": atr_used,
                    },
                )
                orders.append(order)

        return orders

    def _get_levels(
        self, pos: Position, state: StrategyState
    ) -> tuple[float | None, float | None, float | None]:
        """Get or calculate TP/SL levels."""
        # Check if we have stored levels and should use them
        if self.use_entry_atr:
            tp_price = pos.meta.get(self._tp_price_key)
            sl_price = pos.meta.get(self._sl_price_key)
            atr_used = pos.meta.get(self._atr_used_key)
            if tp_price is not None and sl_price is not None:
                return tp_price, sl_price, atr_used

        # Get ATR value
        atr = self._get_atr(pos, state)
        if atr is None or atr <= 0:
            return None, None, None

        # Calculate levels based on position type
        if pos.position_type == PositionType.LONG:
            tp_price = pos.entry_price + atr * self.tp_atr_mult
            sl_price = pos.entry_price - atr * self.sl_atr_mult
        else:
            tp_price = pos.entry_price - atr * self.tp_atr_mult
            sl_price = pos.entry_price + atr * self.sl_atr_mult

        # Store levels if using entry ATR
        if self.use_entry_atr:
            pos.meta[self._tp_price_key] = tp_price
            pos.meta[self._sl_price_key] = sl_price
            pos.meta[self._atr_used_key] = atr

        return tp_price, sl_price, atr

    def _get_atr(self, pos: Position, state: StrategyState) -> float | None:
        """Get ATR value from state or position."""
        # Try state.runtime first (current ATR)
        atr = state.runtime.get("atr", {}).get(pos.pair)
        if atr is not None:
            return atr

        # Fallback to entry ATR stored in position.meta
        return pos.meta.get("entry_atr")
