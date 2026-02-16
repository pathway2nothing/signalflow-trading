from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from signalflow.core import Order, Signals, SignalType, StrategyState, sf_component
from signalflow.core.signal_registry import DIRECTIONAL_SIGNAL_MAP
from signalflow.strategy.component.base import EntryRule


@dataclass
@sf_component(name="fixed_size_entry")
class FixedSizeEntryRule(EntryRule):
    """Simple entry rule with fixed position size.

    Attributes:
        signal_type_map: Mapping signal_type -> "BUY"/"SELL". When set,
            overrides ``signal_types`` for filtering and side determination.
            None = legacy behavior using ``signal_types`` list.
        signal_types: Legacy list of actionable signal types (used when
            signal_type_map is None).
    """

    signal_type_map: dict[str, str] | None = None  # signal_type -> "BUY"/"SELL"

    position_size: float = 0.01
    signal_types: list[str] = field(default_factory=lambda: [SignalType.RISE.value])
    max_positions: int = 10

    pair_col: str = "pair"

    def check_entries(self, signals: Signals, prices: dict[str, float], state: StrategyState) -> list[Order]:
        orders: list[Order] = []

        if signals is None or signals.value.height == 0:
            return orders

        open_count = len(state.portfolio.open_positions())
        if open_count >= self.max_positions:
            return orders

        if self.signal_type_map is not None:
            actionable = list(self.signal_type_map.keys())
        else:
            actionable = self.signal_types

        df = signals.value.filter(pl.col("signal_type").is_in(actionable))

        for row in df.iter_rows(named=True):
            if open_count >= self.max_positions:
                break

            pair = row[self.pair_col]
            signal_type = row["signal_type"]

            price = prices.get(pair)
            if price is None or price <= 0:
                continue

            if self.signal_type_map is not None:
                side = self.signal_type_map.get(signal_type)
                if side is None:
                    continue
            else:
                side = "BUY" if signal_type == SignalType.RISE.value else "SELL"

            order = Order(
                pair=pair, side=side, order_type="MARKET", qty=self.position_size, meta={"signal_type": signal_type}
            )
            orders.append(order)
            open_count += 1

        return orders

    @classmethod
    def from_directional_map(cls, **kwargs) -> FixedSizeEntryRule:
        """Create entry rule using the global DIRECTIONAL_SIGNAL_MAP."""
        return cls(signal_type_map=dict(DIRECTIONAL_SIGNAL_MAP), **kwargs)
