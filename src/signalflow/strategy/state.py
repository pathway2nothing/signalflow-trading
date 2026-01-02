# IMPORTANT

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from signalflow.core.containers import Portfolio


@dataclass(slots=True)
class StrategyState:
    strategy_id: str
    last_ts: datetime | None = None

    runtime: dict[str, Any] = field(default_factory=dict)
    portfolio: Portfolio = field(default_factory=Portfolio)

    last_event_id: str | None = None
