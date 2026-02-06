"""Signal-based entry rule with injectable sizers and filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from signalflow.core import Order, Position, Signals, SignalType, StrategyState, sf_component
from signalflow.strategy.component.base import EntryRule
from signalflow.strategy.component.sizing.base import PositionSizer, SignalContext

if TYPE_CHECKING:
    from signalflow.strategy.component.entry.filters import CompositeEntryFilter, EntryFilter


@dataclass
@sf_component(name="signal", override=True)
class SignalEntryRule(EntryRule):
    """Signal-based entry rule with injectable sizer and filters.

    Converts signals to entry orders with configurable position sizing
    and pre-trade filtering.

    Args:
        position_sizer: Optional PositionSizer for custom sizing logic.
        entry_filters: Optional list of EntryFilters for pre-trade validation.

        # Legacy parameters (for backward compatibility)
        base_position_size: Base notional value (used if no sizer provided).
        use_probability_sizing: Scale size by probability (legacy mode).
        min_probability: Minimum signal probability.
        max_positions_per_pair: Maximum concurrent positions per pair.
        max_total_positions: Maximum total open positions.
        allow_shorts: Allow FALL signals to create short positions.
        max_capital_usage: Maximum fraction of equity in positions.
        min_order_notional: Minimum order size.
        pair_col: Column name for pair in signals.
        ts_col: Column name for timestamp in signals.

    Example:
        >>> # With custom sizer and filters
        >>> entry = SignalEntryRule(
        ...     position_sizer=FixedFractionSizer(fraction=0.02),
        ...     entry_filters=[
        ...         DrawdownFilter(max_drawdown=0.10),
        ...         PriceDistanceFilter(min_distance_pct=0.02),
        ...     ],
        ...     max_positions_per_pair=5,  # For grid strategy
        ... )

        >>> # Legacy mode (backward compatible)
        >>> entry = SignalEntryRule(
        ...     base_position_size=100.0,
        ...     use_probability_sizing=True,
        ... )
    """

    # === New injectable components ===
    position_sizer: PositionSizer | None = None
    entry_filters: list[EntryFilter] | EntryFilter | None = None

    # === Legacy parameters (for backward compatibility) ===
    base_position_size: float = 100.0
    use_probability_sizing: bool = True
    min_probability: float = 0.5
    max_positions_per_pair: int = 1
    max_total_positions: int = 20
    allow_shorts: bool = False
    max_capital_usage: float = 0.95
    min_order_notional: float = 10.0
    pair_col: str = "pair"
    ts_col: str = "timestamp"

    # Internal
    _composite_filter: CompositeEntryFilter | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalize filters to composite."""
        if self.entry_filters is not None:
            # Import here to avoid circular import
            from signalflow.strategy.component.entry.filters import (
                CompositeEntryFilter,
                EntryFilter,
            )

            if isinstance(self.entry_filters, EntryFilter):
                self._composite_filter = CompositeEntryFilter(filters=[self.entry_filters])
            elif isinstance(self.entry_filters, list):
                self._composite_filter = CompositeEntryFilter(filters=self.entry_filters)

    def check_entries(
        self, signals: Signals, prices: dict[str, float], state: StrategyState
    ) -> list[Order]:
        """Check signals and generate entry orders."""
        orders: list[Order] = []

        if signals is None or signals.value.height == 0:
            return orders

        # Build positions map
        positions_by_pair: dict[str, list[Position]] = {}
        for pos in state.portfolio.open_positions():
            positions_by_pair.setdefault(pos.pair, []).append(pos)

        total_open = len(state.portfolio.open_positions())
        if total_open >= self.max_total_positions:
            return orders

        # Calculate capital limits
        available_cash = state.portfolio.cash
        used_capital = sum(pos.entry_price * pos.qty for pos in state.portfolio.open_positions())
        total_equity = available_cash + used_capital
        max_allowed_in_positions = total_equity * self.max_capital_usage
        remaining_allocation = max_allowed_in_positions - used_capital

        # Filter signals
        df = self._filter_signals(signals.value)

        for row in df.iter_rows(named=True):
            if total_open >= self.max_total_positions:
                break
            if remaining_allocation <= self.min_order_notional:
                break
            if available_cash <= self.min_order_notional:
                break

            pair = row[self.pair_col]
            signal_type = row["signal_type"]
            probability = row.get("probability", 1.0) or 1.0

            # Check position limits
            existing_positions = positions_by_pair.get(pair, [])
            if len(existing_positions) >= self.max_positions_per_pair:
                continue

            price = prices.get(pair)
            if price is None or price <= 0:
                continue

            # Determine side
            side = self._determine_side(signal_type)
            if side is None:
                continue

            # Build signal context
            signal_ctx = SignalContext(
                pair=pair,
                signal_type=signal_type,
                probability=probability,
                price=price,
                timestamp=row.get(self.ts_col),
                meta=dict(row),
            )

            # === Apply filters ===
            if self._composite_filter is not None:
                allowed, _reason = self._composite_filter.allow_entry(signal_ctx, state, prices)
                if not allowed:
                    continue

            # === Compute size ===
            if self.position_sizer is not None:
                notional = self.position_sizer.compute_size(signal_ctx, state, prices)
            else:
                # Legacy sizing logic
                notional = self._compute_legacy_size(probability)

            # Apply capital constraints
            notional = min(notional, available_cash * 0.99)
            notional = min(notional, remaining_allocation)

            if notional < self.min_order_notional:
                continue

            qty = notional / price

            order = Order(
                pair=pair,
                side=side,
                order_type="MARKET",
                qty=qty,
                signal_strength=probability,
                meta={
                    "signal_type": signal_type,
                    "signal_probability": probability,
                    "signal_ts": row.get(self.ts_col),
                    "requested_notional": notional,
                    "sizer_used": (
                        self.position_sizer.__class__.__name__
                        if self.position_sizer
                        else "legacy"
                    ),
                },
            )
            orders.append(order)

            # Update tracking
            total_open += 1
            available_cash -= notional * 1.002
            remaining_allocation -= notional
            positions_by_pair.setdefault(pair, []).append(None)

        return orders

    def _filter_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter signals by type and probability."""
        actionable_types = [SignalType.RISE.value]
        if self.allow_shorts:
            actionable_types.append(SignalType.FALL.value)

        df = df.filter(pl.col("signal_type").is_in(actionable_types))

        if "probability" in df.columns:
            df = df.filter(pl.col("probability") >= self.min_probability)
            df = df.sort("probability", descending=True)

        return df

    def _determine_side(self, signal_type: str) -> str | None:
        """Determine order side from signal type."""
        if signal_type == SignalType.RISE.value:
            return "BUY"
        elif signal_type == SignalType.FALL.value and self.allow_shorts:
            return "SELL"
        return None

    def _compute_legacy_size(self, probability: float) -> float:
        """Legacy sizing for backward compatibility."""
        notional = self.base_position_size
        if self.use_probability_sizing:
            notional *= probability
        return notional
