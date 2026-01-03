# src/signalflow/data/strategy_store/duckdb.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Iterable, Optional

import duckdb

from signalflow.core import StrategyState, Position, Trade

from signalflow.data.strategy_store.base import StrategyStore
from signalflow.data.strategy_store.schema import SCHEMA_SQL


def _to_json(obj) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, default=str, ensure_ascii=False)


def _state_from_json(payload: str) -> StrategyState:
    data = json.loads(payload)
    return StrategyState(**data)


class DuckDbStrategyStore(StrategyStore):
    def __init__(self, path: str) -> None:
        self.path = path
        self.con = duckdb.connect(path)

    def init(self) -> None:
        self.con.execute(SCHEMA_SQL)

    def load_state(self, strategy_id: str) -> Optional[StrategyState]:
        row = self.con.execute(
            "SELECT payload_json FROM strategy_state WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()
        if not row:
            return None
        return _state_from_json(row[0])

    def save_state(self, state: StrategyState) -> None:
        payload = _to_json(state)
        self.con.execute(
            """
            INSERT INTO strategy_state(strategy_id, last_ts, last_event_id, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET
              last_ts = excluded.last_ts,
              last_event_id = excluded.last_event_id,
              payload_json = excluded.payload_json
            """,
            [state.strategy_id, state.last_ts, state.last_event_id, payload],
        )

    def upsert_positions(self, strategy_id: str, ts: datetime, positions: Iterable[Position]) -> None:
        rows = []
        for p in positions:
            pid = getattr(p, "id", None)
            if pid is None:
                raise ValueError("Position must have id")
            rows.append((strategy_id, ts, str(pid), _to_json(p)))

        if not rows:
            return

        self.con.executemany(
            """
            INSERT INTO positions(strategy_id, ts, position_id, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(strategy_id, ts, position_id) DO UPDATE SET
              payload_json = excluded.payload_json
            """,
            rows,
        )

    def append_trade(self, strategy_id: str, trade: Trade) -> None:
        tid = getattr(trade, "id", None) or getattr(trade, "trade_id", None)
        ts = getattr(trade, "ts", None) or getattr(trade, "timestamp", None)
        if tid is None or ts is None:
            raise ValueError("Trade must have id and ts/timestamp")

        self.con.execute(
            """
            INSERT INTO trades(strategy_id, ts, trade_id, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(strategy_id, trade_id) DO NOTHING
            """,
            [strategy_id, ts, str(tid), _to_json(trade)],
        )

    def append_metrics(self, strategy_id: str, ts: datetime, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        rows = [(strategy_id, ts, k, float(v)) for k, v in metrics.items()]
        self.con.executemany(
            """
            INSERT INTO metrics(strategy_id, ts, name, value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(strategy_id, ts, name) DO UPDATE SET value = excluded.value
            """,
            rows,
        )
