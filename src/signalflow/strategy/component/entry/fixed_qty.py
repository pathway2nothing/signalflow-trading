# IMPORTANT

from typing import Any, ClassVar

from signalflow.core import sf_component
from signalflow.core.enums import PositionType
from signalflow.core.signals import Signals

from signalflow.strategy.component.base import StrategyEntryRule
from signalflow.strategy.types import NewPositionIntent, NewPositionOrder, StrategyContext
from signalflow.strategy.state import StrategyState


@sf_component(name="fixed_qty")
class FixedQtyEntryRule(StrategyEntryRule):
    def __init__(self, *, qty: float = 1.0, max_positions: int = 10) -> None:
        self.qty = float(qty)
        self.max_positions = int(max_positions)

    def build_intents(self, *, signals: Signals, state: StrategyState, context: StrategyContext) -> list[NewPositionIntent]:
        df = signals if hasattr(signals, "to_polars") is False else signals.to_polars()
        if not hasattr(df, "filter"):
            return []

        if len(state.portfolio.open_positions()) >= self.max_positions:
            return []

        rows = (
            df.filter((df["timestamp"] == context.ts) & (df["signal"] != 0))
              .select(["pair", "signal"])
              .to_dicts()
        )

        intents: list[NewPositionIntent] = []
        for r in rows:
            side = PositionType.LONG if r["signal"] > 0 else PositionType.SHORT
            intents.append(NewPositionIntent(pair=r["pair"], position_type=side, ts=context.ts))
        return intents

    def size_intents(self, *, intents: list[NewPositionIntent], state: StrategyState, context: StrategyContext) -> list[NewPositionOrder]:
        out: list[NewPositionOrder] = []
        for i in intents:
            out.append(NewPositionOrder(
                pair=i.pair,
                position_type=i.position_type,
                ts=i.ts,
                qty=self.qty,
                signal_strength=i.signal_strength,
                meta=i.meta,
            ))
        return out

    def build_orders(self, *, signals: Signals, state: StrategyState, context: StrategyContext) -> list[NewPositionOrder]:
        return self.size_intents(intents=self.build_intents(signals=signals, state=state, context=context), state=state, context=context)
