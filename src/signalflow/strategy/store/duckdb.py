import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from signalflow.core.containers import Position, Trade, Portfolio
from signalflow.core.enums import PositionType
from signalflow.strategy.journal.base import StrategyJournal 

def _dumps(x: dict[str, Any]) -> str:
    return json.dumps(x or {}, ensure_ascii=False, separators=(",", ":"))


def _loads(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


@dataclass
class DuckDbStrategyStore(StrategyJournal):
    db_path: Path
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._con = duckdb.connect(str(self.db_path))
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self._con.execute("""
        CREATE TABLE IF NOT EXISTS sf_portfolio (
          strategy_id VARCHAR PRIMARY KEY,
          cash DOUBLE NOT NULL,
          meta_json VARCHAR
        );
        """)

        self._con.execute("""
        CREATE TABLE IF NOT EXISTS sf_positions (
          strategy_id VARCHAR NOT NULL,
          id VARCHAR NOT NULL,
          is_closed BOOLEAN NOT NULL,
          pair VARCHAR NOT NULL,
          position_type VARCHAR NOT NULL,
          signal_strength DOUBLE NOT NULL,
          entry_time TIMESTAMP,
          last_time TIMESTAMP,
          entry_price DOUBLE NOT NULL,
          last_price DOUBLE NOT NULL,
          qty DOUBLE NOT NULL,
          fees_paid DOUBLE NOT NULL,
          realized_pnl DOUBLE NOT NULL,
          meta_json VARCHAR,
          PRIMARY KEY(strategy_id, id)
        );
        """)
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_sf_pos_open ON sf_positions(strategy_id, is_closed);")

        self._con.execute("""
        CREATE TABLE IF NOT EXISTS sf_trades (
          strategy_id VARCHAR NOT NULL,
          id VARCHAR NOT NULL,
          position_id VARCHAR,
          pair VARCHAR NOT NULL,
          side VARCHAR NOT NULL,
          ts TIMESTAMP,
          price DOUBLE NOT NULL,
          qty DOUBLE NOT NULL,
          fee DOUBLE NOT NULL,
          meta_json VARCHAR,
          PRIMARY KEY(strategy_id, id)
        );
        """)
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_sf_trades_ts ON sf_trades(strategy_id, ts);")

        self._con.execute("""
        CREATE TABLE IF NOT EXISTS sf_metrics (
          strategy_id VARCHAR NOT NULL,
          ts TIMESTAMP NOT NULL,
          key VARCHAR NOT NULL,
          value DOUBLE,
          PRIMARY KEY(strategy_id, ts, key)
        );
        """)
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_sf_metrics_ts ON sf_metrics(strategy_id, ts);")

        self._con.execute("""
        CREATE TABLE IF NOT EXISTS sf_kv (
          strategy_id VARCHAR NOT NULL,
          key VARCHAR NOT NULL,
          value VARCHAR,
          PRIMARY KEY(strategy_id, key)
        );
        """)

    def close(self) -> None:
        self._con.close()

    def load_portfolio(self, strategy_id: str) -> Portfolio:
        row = self._con.execute(
            "SELECT cash, meta_json FROM sf_portfolio WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()
        if row is None:
            return Portfolio(cash=0.0)
        cash, meta_json = row
        p = Portfolio(cash=float(cash))
        return p

    def save_portfolio(self, strategy_id: str, portfolio: Portfolio) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO sf_portfolio(strategy_id, cash, meta_json) VALUES (?, ?, ?)",
            [strategy_id, float(portfolio.cash), _dumps({})],
        )

    def load_open_positions(self, strategy_id: str) -> list[Position]:
        rows = self._con.execute(
            """
            SELECT id, is_closed, pair, position_type, signal_strength,
                   entry_time, last_time, entry_price, last_price,
                   qty, fees_paid, realized_pnl, meta_json
            FROM sf_positions
            WHERE strategy_id = ? AND is_closed = FALSE
            """,
            [strategy_id],
        ).fetchall()

        out: list[Position] = []
        for r in rows:
            (
                pid, is_closed, pair, pt, ss,
                entry_time, last_time, entry_price, last_price,
                qty, fees_paid, realized_pnl, meta_json
            ) = r

            out.append(Position(
                id=str(pid),
                is_closed=bool(is_closed),
                pair=str(pair),
                position_type=PositionType(str(pt)),
                signal_strength=float(ss),
                entry_time=entry_time,
                last_time=last_time,
                entry_price=float(entry_price),
                last_price=float(last_price),
                qty=float(qty),
                fees_paid=float(fees_paid),
                realized_pnl=float(realized_pnl),
                meta=_loads(meta_json),
            ))
        return out

    def upsert_positions(self, strategy_id: str, positions: list[Position]) -> None:
        if not positions:
            return
        values = [
            (
                strategy_id,
                p.id,
                bool(p.is_closed),
                p.pair,
                p.position_type.value,
                float(p.signal_strength),
                p.entry_time,
                p.last_time,
                float(p.entry_price),
                float(p.last_price),
                float(p.qty),
                float(p.fees_paid),
                float(p.realized_pnl),
                _dumps(p.meta),
            )
            for p in positions
        ]
        self._con.executemany(
            """
            INSERT OR REPLACE INTO sf_positions(
              strategy_id, id, is_closed, pair, position_type, signal_strength,
              entry_time, last_time, entry_price, last_price,
              qty, fees_paid, realized_pnl, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

    def close_positions(self, strategy_id: str, position_ids: list[str], ts: datetime, reason: str) -> None:
        if not position_ids:
            return
        placeholders = ",".join(["?"] * len(position_ids))
        self._con.execute(
            f"""
            UPDATE sf_positions
            SET is_closed = TRUE, last_time = ?
            WHERE strategy_id = ? AND id IN ({placeholders})
            """,
            [ts, strategy_id, *position_ids],
        )
        self.kv_set(strategy_id, f"close_reason:{ts.isoformat()}", reason)


    def insert_trades(self, strategy_id: str, trades: list[Trade]) -> None:
        if not trades:
            return
        values = [
            (
                strategy_id,
                t.id,
                t.position_id,
                t.pair,
                t.side,
                t.ts,
                float(t.price),
                float(t.qty),
                float(t.fee),
                _dumps(t.meta),
            )
            for t in trades
        ]
        self._con.executemany(
            """
            INSERT OR REPLACE INTO sf_trades(
              strategy_id, id, position_id, pair, side, ts, price, qty, fee, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

    def insert_metrics(self, strategy_id: str, ts: datetime, values: dict[str, float]) -> None:
        if not values:
            return
        rows = [(strategy_id, ts, k, float(v)) for k, v in values.items()]
        self._con.executemany(
            "INSERT OR REPLACE INTO sf_metrics(strategy_id, ts, key, value) VALUES (?, ?, ?, ?)",
            rows,
        )

    def kv_get(self, strategy_id: str, key: str) -> str | None:
        r = self._con.execute(
            "SELECT value FROM sf_kv WHERE strategy_id = ? AND key = ?",
            [strategy_id, key],
        ).fetchone()
        return None if r is None else r[0]

    def kv_set(self, strategy_id: str, key: str, value: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO sf_kv(strategy_id, key, value) VALUES (?, ?, ?)",
            [strategy_id, key, value],
        )
