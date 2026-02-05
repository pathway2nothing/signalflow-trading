from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Iterable, Optional

from signalflow.core import sf_component, StrategyState, Position, Trade

from signalflow.data.strategy_store.base import StrategyStore
from signalflow.data.strategy_store.schema import SCHEMA_SQL
from signalflow.data.strategy_store._serialization import to_json as _to_json, state_from_json as _state_from_json


def _adapt_datetime(val: datetime) -> str:
    return val.isoformat(" ")


def _convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_datetime)


@sf_component(name="sqlite/strategy")
class SqliteStrategyStore(StrategyStore):
    """SQLite implementation of strategy persistence. Zero extra deps."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.con = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self.con.execute("PRAGMA journal_mode=WAL")

    def init(self) -> None:
        self.con.executescript(SCHEMA_SQL)
        self.con.commit()

    def load_state(self, strategy_id: str) -> Optional[StrategyState]:
        row = self.con.execute(
            "SELECT payload_json FROM strategy_state WHERE strategy_id = ?",
            (strategy_id,),
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
            (state.strategy_id, state.last_ts, state.last_event_id, payload),
        )
        self.con.commit()

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
        self.con.commit()

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
            (strategy_id, ts, str(tid), _to_json(trade)),
        )
        self.con.commit()

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
        self.con.commit()

    def close(self) -> None:
        self.con.close()
