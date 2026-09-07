"""Risk layer - deterministic hard constraints on proposed intents."""

import os
from dataclasses import dataclass, replace

from loguru import logger

from signalflow.engine.types import Intent, PortfolioSnapshot
from signalflow.enums import IntentKind
from signalflow.errors import KillSwitchTripped


@dataclass
class Risk:
    """Clip intents against drawdown, position, and notional limits.

    ``max_positions`` caps concurrent open positions global across pairs. The kill
    switch drops every new entry (closes still pass): it engages on a drawdown
    breach or via :meth:`trip`, and - when ``kill_switch_path`` is set - whenever
    that file exists, re-checked on every :meth:`clip`, so an operator can stop or
    release a running loop from outside. :meth:`state`/:meth:`restore` let the live
    loop persist it across restarts.
    """

    max_drawdown: float = 1.0
    max_positions: int = 1_000
    max_notional_per_pair: float = 1.0
    kill_switch_path: str | None = None

    def __post_init__(self) -> None:
        self._tripped = False
        self.reason = ""
        self._file_seen = self._file_tripped()
        if self._file_seen:
            logger.warning(f"risk kill switch is engaged at start (file {self.kill_switch_path!r} exists)")

    def _file_tripped(self) -> bool:
        return bool(self.kill_switch_path) and os.path.exists(self.kill_switch_path)

    @property
    def tripped(self) -> bool:
        """Engaged now. With ``kill_switch_path`` the file is the source of truth and is re-read on
        every call, so an operator can trip or release a running loop by creating or deleting it."""
        if not self.kill_switch_path:
            return self._tripped
        now = self._file_tripped()
        if now != self._file_seen:
            if now:
                logger.warning(f"risk kill switch ENGAGED: file {self.kill_switch_path!r} appeared")
            else:
                logger.warning(f"risk kill switch RELEASED: file {self.kill_switch_path!r} removed")
            self._file_seen = now
        return now

    def trip(self, reason: str = "") -> None:
        """Engage the kill switch (and write the file when a path is configured)."""
        if not self.tripped:
            logger.warning(f"risk kill switch TRIPPED: {reason}")
        self._tripped = True
        self.reason = reason or "tripped"
        if self.kill_switch_path:
            with open(self.kill_switch_path, "w") as fh:
                fh.write(self.reason)
            self._file_seen = True

    def reset(self) -> None:
        """Release the kill switch explicitly (and remove the file when a path is configured)."""
        if self.tripped:
            logger.warning("risk kill switch RESET")
        self._tripped = False
        self.reason = ""
        if self.kill_switch_path and os.path.exists(self.kill_switch_path):
            os.remove(self.kill_switch_path)
        self._file_seen = False

    def state(self) -> dict:
        """What ``save_state`` persists so a restart resumes tripped when it was tripped."""
        return {"tripped": bool(self.tripped), "reason": self.reason}

    def restore(self, state: "dict | None") -> None:
        """Resume from :meth:`state`; the kill-switch file, when configured, still wins."""
        if not state:
            return
        if state.get("tripped") and not self.tripped:
            self._tripped = True
            self.reason = str(state.get("reason") or "restored")
            logger.warning(f"risk kill switch restored as TRIPPED from saved state: {self.reason}")
            if self.kill_switch_path:
                with open(self.kill_switch_path, "w") as fh:
                    fh.write(self.reason)
                self._file_seen = True

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

        tripped = self.tripped
        if tripped and raise_on_trip:
            raise KillSwitchTripped(f"kill switch engaged; refusing to send orders (path={self.kill_switch_path!r})")

        out: list[Intent] = []
        n_pos = len(portfolio.positions)
        for it in intents:
            if it.kind == IntentKind.CLOSE:
                out.append(it)
                n_pos = max(0, n_pos - 1)
                continue
            if tripped:
                continue
            if n_pos >= self.max_positions:
                continue
            if it.notional is not None:
                it = replace(it, notional=min(it.notional, self.max_notional_per_pair * eq))
            out.append(it)
            n_pos += 1
        return out
