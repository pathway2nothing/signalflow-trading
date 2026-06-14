"""RulesStrategy - entry/exit/sizing rules."""


from dataclasses import dataclass, field

from signalflow.decorators import strategy
from signalflow.engine.types import Intent
from signalflow.enums import FALL, RISE, IntentKind, Side
from signalflow.strategy.observation import Observation


@dataclass
class Entry:
    """Position entry policy."""

    size_pct: float = 0.1
    min_p_success: float = 0.0
    max_positions: int = 1_000


@dataclass
class Exit:
    """Take-profit / stop-loss exit policy."""

    tp: float = 0.03
    sl: float = 0.015


@strategy("rules")
@dataclass
class RulesStrategy:
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
        opened: set[str] = set()
        for row in sig.to_dicts():
            pair = row["pair"]
            s = row.get("signal")
            if pair in held:
                if s == FALL and pair not in closing:
                    intents.append(
                        Intent(pair, IntentKind.CLOSE, Side.SELL, qty=held[pair].qty, reason="fall_exit")
                    )
                    closing.add(pair)
                continue
            if pair in opened:
                continue
            if s == RISE and n_open < self.entry.max_positions:
                p_succ = row.get("p_success")
                if p_succ is not None and p_succ < self.entry.min_p_success:
                    continue
                notional = self.entry.size_pct * port.equity
                if notional > 0:
                    intents.append(Intent(pair, IntentKind.OPEN, Side.BUY, notional=notional, reason="signal"))
                    opened.add(pair)
                    n_open += 1
        return intents
