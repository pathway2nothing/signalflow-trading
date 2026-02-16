from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from signalflow.core import Position, StrategyState, Trade, sf_component
from signalflow.data.strategy_store._serialization import state_from_json as _state_from_json
from signalflow.data.strategy_store._serialization import to_json as _to_json
from signalflow.data.strategy_store.base import StrategyStore
from signalflow.data.strategy_store.schema import PG_SCHEMA_SQL

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore[assignment]

_PG_MISSING_MSG = "psycopg is required for PostgreSQL stores. Install with: pip install signalflow-trading[postgres]"


@sf_component(name="postgres/strategy")
class PgStrategyStore(StrategyStore):
    """PostgreSQL implementation of strategy persistence.

    Requires psycopg. Install with: pip install signalflow-trading[postgres]
    Connection string from dsn param or SIGNALFLOW_PG_DSN env var.
    """

    def __init__(self, dsn: str = "", **kwargs: Any) -> None:
        if psycopg is None:
            raise ImportError(_PG_MISSING_MSG)
        resolved_dsn = dsn or os.environ.get("SIGNALFLOW_PG_DSN", "")
        if not resolved_dsn:
            raise ValueError("PostgreSQL DSN must be provided via dsn param or SIGNALFLOW_PG_DSN env var")
        self.dsn = resolved_dsn
        self.con = psycopg.connect(resolved_dsn, autocommit=False)

    def init(self) -> None:
        with self.con.cursor() as cur:
            cur.execute(PG_SCHEMA_SQL)
        self.con.commit()

    def load_state(self, strategy_id: str) -> StrategyState | None:
        with self.con.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM strategy_state WHERE strategy_id = %s",
                (strategy_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return _state_from_json(row[0])

    def save_state(self, state: StrategyState) -> None:
        payload = _to_json(state)
        with self.con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO strategy_state(strategy_id, last_ts, last_event_id, payload_json)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(strategy_id) DO UPDATE SET
                  last_ts = EXCLUDED.last_ts,
                  last_event_id = EXCLUDED.last_event_id,
                  payload_json = EXCLUDED.payload_json
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

        with self.con.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO positions(strategy_id, ts, position_id, payload_json)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(strategy_id, ts, position_id) DO UPDATE SET
                  payload_json = EXCLUDED.payload_json
                """,
                rows,
            )
        self.con.commit()

    def append_trade(self, strategy_id: str, trade: Trade) -> None:
        tid = getattr(trade, "id", None) or getattr(trade, "trade_id", None)
        ts = getattr(trade, "ts", None) or getattr(trade, "timestamp", None)
        if tid is None or ts is None:
            raise ValueError("Trade must have id and ts/timestamp")

        with self.con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trades(strategy_id, ts, trade_id, payload_json)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(strategy_id, trade_id) DO NOTHING
                """,
                (strategy_id, ts, str(tid), _to_json(trade)),
            )
        self.con.commit()

    def append_metrics(self, strategy_id: str, ts: datetime, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        rows = [(strategy_id, ts, k, float(v)) for k, v in metrics.items()]
        with self.con.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO metrics(strategy_id, ts, name, value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(strategy_id, ts, name) DO UPDATE SET value = EXCLUDED.value
                """,
                rows,
            )
        self.con.commit()

    def close(self) -> None:
        self.con.close()
