"""Risk layer - deterministic hard constraints on proposed intents."""

import os
from dataclasses import dataclass

from loguru import logger

from signalflow.engine.types import Intent, PortfolioSnapshot
from signalflow.enums import IntentKind
from signalflow.errors import KillSwitchTripped


@dataclass
class Risk:
    """Clip intents against drawdown, position, and notional limits.

    ``max_positions`` caps concurrent open positions global across pairs.
    """

    max_drawdown: float = 1.0
    max_positions: int = 1_000
    max_notional_per_pair: float = 1.0
    kill_switch_path: str | None = None

    def __post_init__(self) -> None:
        self._tripped = self._load_tripped()

    def _load_tripped(self) -> bool:
        return bool(self.kill_switch_path) and os.path.exists(self.kill_switch_path)

    @property
    def tripped(self) -> bool:
        return self._tripped

    def trip(self, reason: str = "") -> None:
        if not self._tripped:
            logger.warning(f"risk kill switch TRIPPED: {reason}")
        self._tripped = True
        if self.kill_switch_path:
            with open(self.kill_switch_path, "w") as fh:
                fh.write(reason or "tripped")

    def reset(self) -> None:
        self._tripped = False
        if self.kill_switch_path and os.path.exists(self.kill_switch_path):
            os.remove(self.kill_switch_path)

    def clip(
        self,
        intents: list[Intent],
        portfolio: PortfolioSnapshot,
        peak_equity: float,
        raise_on_trip: bool = False,
    ) -> list[Intent]:
        """Clip intents; with ``raise_on_trip`` a tripped kill switch halts loudly instead of dropping."""
        eq = portfolio.equity
        if peak_equity > 0 and (peak_equity - eq) / peak_equity >= self.max_drawdown:
            self.trip(f"drawdown {(peak_equity - eq) / peak_equity:.3f} >= {self.max_drawdown}")

        if self._tripped and raise_on_trip:
            raise KillSwitchTripped(f"kill switch engaged; refusing to send orders (path={self.kill_switch_path!r})")

        out: list[Intent] = []
        n_pos = len(portfolio.positions)
        for it in intents:
            if it.kind == IntentKind.CLOSE:
                out.append(it)
                n_pos = max(0, n_pos - 1)
                continue
            if self._tripped:
                continue
            if n_pos >= self.max_positions:
                continue
            if it.notional is not None:
                it.notional = min(it.notional, self.max_notional_per_pair * eq)
            out.append(it)
            n_pos += 1
        return out
