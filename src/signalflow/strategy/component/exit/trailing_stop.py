from dataclasses import dataclass

from signalflow.core import Order, Position, PositionType, StrategyState, sf_component
from signalflow.strategy.component.base import ExitRule


@dataclass
@sf_component(name="trailing_stop")
class TrailingStopExit(ExitRule):
    """Trailing stop exit rule.

    Tracks the highest price (LONG) or lowest price (SHORT) since entry
    and exits when price retraces by trail_pct from the peak/trough.

    Optionally, can activate only after reaching activation_pct profit.

    Args:
        trail_pct: Trailing distance as percentage (e.g., 0.02 = 2%).
        activation_pct: Start trailing only after X% profit. None = immediate.
        use_atr: If True, use ATR-based trailing distance instead of percentage.
        atr_multiplier: Trail distance = ATR * multiplier (when use_atr=True).

    Example:
        >>> rule = TrailingStopExit(trail_pct=0.03, activation_pct=0.02)
        >>> # Starts trailing after 2% profit, trails at 3% from peak
    """

    trail_pct: float = 0.02
    activation_pct: float | None = None
    use_atr: bool = False
    atr_multiplier: float = 2.0

    # Internal meta keys
    _peak_key: str = "_trailing_peak"
    _trough_key: str = "_trailing_trough"
    _activated_key: str = "_trailing_activated"

    def check_exits(self, positions: list[Position], prices: dict[str, float], state: StrategyState) -> list[Order]:
        orders: list[Order] = []

        for pos in positions:
            if pos.is_closed:
                continue

            price = prices.get(pos.pair)
            if price is None or price <= 0:
                continue

            # Check activation
            if not self._check_activation(pos, price):
                continue

            # Update tracking price and check exit
            if pos.position_type == PositionType.LONG:
                order = self._check_long_exit(pos, price, state)
            else:
                order = self._check_short_exit(pos, price, state)

            if order:
                orders.append(order)

        return orders

    def _check_activation(self, pos: Position, price: float) -> bool:
        """Check if trailing is activated."""
        if self.activation_pct is None:
            return True

        if pos.meta.get(self._activated_key, False):
            return True

        # Check if activation threshold reached
        if pos.position_type == PositionType.LONG:
            activated = price >= pos.entry_price * (1 + self.activation_pct)
        else:
            activated = price <= pos.entry_price * (1 - self.activation_pct)

        if activated:
            pos.meta[self._activated_key] = True

        return activated

    def _check_long_exit(self, pos: Position, price: float, state: StrategyState) -> Order | None:
        """Check trailing stop for LONG position."""
        # Update peak
        current_peak = pos.meta.get(self._peak_key, pos.entry_price)
        if price > current_peak:
            pos.meta[self._peak_key] = price
            current_peak = price

        # Calculate trail distance
        trail_distance = self._get_trail_distance(pos, current_peak, state)

        # Check exit condition
        trail_price = current_peak - trail_distance
        if price <= trail_price:
            return self._create_exit_order(pos, price, current_peak)

        return None

    def _check_short_exit(self, pos: Position, price: float, state: StrategyState) -> Order | None:
        """Check trailing stop for SHORT position."""
        # Update trough
        current_trough = pos.meta.get(self._trough_key, pos.entry_price)
        if price < current_trough:
            pos.meta[self._trough_key] = price
            current_trough = price

        # Calculate trail distance
        trail_distance = self._get_trail_distance(pos, current_trough, state)

        # Check exit condition
        trail_price = current_trough + trail_distance
        if price >= trail_price:
            return self._create_exit_order(pos, price, current_trough)

        return None

    def _get_trail_distance(self, pos: Position, reference_price: float, state: StrategyState) -> float:
        """Calculate trailing distance (percentage or ATR-based)."""
        if self.use_atr:
            atr = state.runtime.get("atr", {}).get(pos.pair)
            if atr is not None and atr > 0:
                return atr * self.atr_multiplier
            # Fallback to entry ATR if available
            entry_atr = pos.meta.get("entry_atr")
            if entry_atr is not None and entry_atr > 0:
                return entry_atr * self.atr_multiplier
            # Final fallback to percentage
        return reference_price * self.trail_pct

    def _create_exit_order(self, pos: Position, price: float, peak_or_trough: float) -> Order:
        """Create exit order with metadata."""
        side = "SELL" if pos.position_type == PositionType.LONG else "BUY"
        meta_key = "peak_price" if pos.position_type == PositionType.LONG else "trough_price"

        return Order(
            pair=pos.pair,
            side=side,
            order_type="MARKET",
            qty=pos.qty,
            position_id=pos.id,
            meta={
                "exit_reason": "trailing_stop",
                "entry_price": pos.entry_price,
                "exit_price": price,
                meta_key: peak_or_trough,
                "trail_pct": self.trail_pct,
            },
        )
