"""Clock - injected time source."""

import time
from dataclasses import dataclass

from signalflow.enums import RunMode


@dataclass
class Clock:
    mode: RunMode = RunMode.BACKTEST

    @property
    def is_live(self) -> bool:
        return self.mode in (RunMode.PAPER, RunMode.LIVE)

    def now(self):
        return time.time() if self.is_live else None

    def wall(self) -> float:
        """Real wall-clock epoch seconds regardless of mode (for scheduling)."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds`` (clamped at zero)."""
        time.sleep(max(0.0, seconds))
