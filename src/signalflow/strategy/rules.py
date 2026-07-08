"""RulesStrategy - entry/exit/sizing rules."""

from dataclasses import dataclass, field

from signalflow.decorators import strategy
from signalflow.engine.types import Intent
from signalflow.enums import FALL, RISE, IntentKind, Side
from signalflow.strategy.base import Strategy
from signalflow.strategy.observation import Observation


@dataclass
class Entry:
    """Position entry policy.

    When a validator is present (the signal frame carries a ``p_success`` column)
    and ``min_p_success`` is positive, a null ``p_success`` counts as not validated
    and the entry is skipped. Without a validator column, entries open unchanged.

    ``max_positions`` caps concurrent open positions *global across pairs*.
    ``max_positions_per_pair`` (when set) caps them *per pair*; ``None`` disables it.
    """

    size_pct: float = 0.1
    min_p_success: float = 0.0
    max_positions: int = 1_000
    max_positions_per_pair: "int | None" = None


@dataclass
class Exit:
    """Take-profit / stop-loss exit policy."""

    tp: float = 0.03
    sl: float = 0.015


@strategy("rules")
@dataclass
class RulesStrategy(Strategy):
    """Turn validated signals into open/close intents via fixed rules."""

    entry: Entry = field(default_factory=Entry)
    exit: Exit = field(default_factory=Exit)

    def decide(self, obs: Observation) -> list[Intent]:
        intents: list[Intent] = []
        port = obs.portfolio
        held = port.positions

        for pair, pos in held.items():
            price = port.prices.get(pair)
            if price is None:
                continue
            ret = pos.return_pct(price)
            if ret >= self.exit.tp or ret <= -self.exit.sl:
                intents.append(Intent(pair, IntentKind.CLOSE, Side.SELL, qty=pos.qty, reason="tp_sl"))

        closing = {i.pair for i in intents}
        n_open = len(held) - len(closing)

        sig = obs.signals
        if "signal" not in sig.columns:
            return intents
        has_validator = "p_success" in sig.columns
        per_pair_cap = self.entry.max_positions_per_pair
        per_pair_open: dict[str, int] = {}
        for pair in held:
            per_pair_open[pair] = per_pair_open.get(pair, 0) + 1
        opened: set[str] = set()
        for row in sig.to_dicts():
            pair = row["pair"]
            s = row.get("signal")
            if pair in held:
                if s == FALL and pair not in closing:
                    intents.append(Intent(pair, IntentKind.CLOSE, Side.SELL, qty=held[pair].qty, reason="fall_exit"))
                    closing.add(pair)
                continue
            if pair in opened:
                continue
            if s == RISE and n_open < self.entry.max_positions:
                if per_pair_cap is not None and per_pair_open.get(pair, 0) >= per_pair_cap:
                    continue
                p_succ = row.get("p_success")
                if has_validator and self.entry.min_p_success > 0 and p_succ is None:
                    continue
                if p_succ is not None and p_succ < self.entry.min_p_success:
                    continue
                notional = self.entry.size_pct * port.equity
                if notional > 0:
                    intents.append(Intent(pair, IntentKind.OPEN, Side.BUY, notional=notional, reason="signal"))
                    opened.add(pair)
                    per_pair_open[pair] = per_pair_open.get(pair, 0) + 1
                    n_open += 1
        return intents
