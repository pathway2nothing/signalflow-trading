from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from signalflow.core.containers import Portfolio


@dataclass(slots=True)
class StrategyState:
    """
    Single source of in-memory strategy state.

    Lives in `signalflow.core` so both `signalflow.strategy` (logic/execution)
    and `signalflow.data` (persistence) can depend on it without cycles.

    Notes:
    - `portfolio` is the canonical portfolio state at the end of the latest processed tick.
    - `runtime` is a flexible bag for cooldowns, watermarks, guards, etc.
    - `metrics` is the latest computed metrics snapshot (optional but practical).
    - `last_event_id` is useful for live idempotency / resume (optional).
    """

    strategy_id: str

    last_ts: Optional[datetime] = None
    last_event_id: Optional[str] = None

    portfolio: Portfolio = field(default_factory=Portfolio)

    runtime: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    metrics_phase_done: set[str] = field(default_factory=set)

    def touch(self, ts: datetime, event_id: Optional[str] = None) -> None:
        """Update watermarks after a successful tick commit."""
        self.last_ts = ts
        if event_id is not None:
            self.last_event_id = event_id

    def reset_tick_cache(self) -> None:
        """Call at the start of every tick if you use phase-gated metrics."""
        self.metrics_phase_done.clear()
