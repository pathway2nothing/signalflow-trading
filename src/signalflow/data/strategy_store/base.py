# src/signalflow/data/strategy_store/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional

from signalflow.core import StrategyState, Position, Trade


class StrategyStore(ABC):
    """Persistence only: load/save state, append events, write metrics."""

    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def load_state(self, strategy_id: str) -> Optional[StrategyState]: ...

    @abstractmethod
    def save_state(self, state: StrategyState) -> None: ...

    @abstractmethod
    def upsert_positions(self, strategy_id: str, ts: datetime, positions: Iterable[Position]) -> None: ...

    @abstractmethod
    def append_trade(self, strategy_id: str, trade: Trade) -> None: ...

    @abstractmethod
    def append_metrics(self, strategy_id: str, ts: datetime, metrics: dict[str, float]) -> None: ...
