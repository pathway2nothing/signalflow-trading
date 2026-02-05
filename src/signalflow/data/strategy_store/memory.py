from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from signalflow.core import sf_component, StrategyState, Position, Trade
from signalflow.data.strategy_store.base import StrategyStore
from signalflow.data.strategy_store._serialization import (
    to_json as _to_json,
    state_from_json as _state_from_json,
)


@sf_component(name="memory/strategy")
class InMemoryStrategyStore(StrategyStore):
    """In-memory implementation of strategy persistence.

    All data lives in plain dicts/lists. Useful for tests, notebooks,
    and short-lived pipelines where persistence is not needed.
    """

    def __init__(self, **kwargs: object) -> None:
        self._states: dict[str, str] = {}  # strategy_id -> payload_json
        self._positions: dict[tuple[str, datetime, str], str] = {}  # (sid, ts, pid) -> json
        self._trades: dict[tuple[str, str], str] = {}  # (sid, tid) -> json
        self._metrics: dict[tuple[str, datetime, str], float] = {}  # (sid, ts, name) -> value

    def init(self) -> None:
        pass  # nothing to initialise

    def load_state(self, strategy_id: str) -> Optional[StrategyState]:
        payload = self._states.get(strategy_id)
        if payload is None:
            return None
        return _state_from_json(payload)

    def save_state(self, state: StrategyState) -> None:
        self._states[state.strategy_id] = _to_json(state)

    def upsert_positions(self, strategy_id: str, ts: datetime, positions: Iterable[Position]) -> None:
        for p in positions:
            pid = getattr(p, "id", None)
            if pid is None:
                raise ValueError("Position must have id")
            self._positions[(strategy_id, ts, str(pid))] = _to_json(p)

    def append_trade(self, strategy_id: str, trade: Trade) -> None:
        tid = getattr(trade, "id", None) or getattr(trade, "trade_id", None)
        ts = getattr(trade, "ts", None) or getattr(trade, "timestamp", None)
        if tid is None or ts is None:
            raise ValueError("Trade must have id and ts/timestamp")
        key = (strategy_id, str(tid))
        if key not in self._trades:
            self._trades[key] = _to_json(trade)

    def append_metrics(self, strategy_id: str, ts: datetime, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self._metrics[(strategy_id, ts, name)] = float(value)

    def close(self) -> None:
        self._states.clear()
        self._positions.clear()
        self._trades.clear()
        self._metrics.clear()
