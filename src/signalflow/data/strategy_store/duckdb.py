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
    """Convert object to JSON string.

    Handles dataclasses by converting to dict first. Uses default=str
    for non-serializable types (e.g., datetime).

    Args:
        obj: Object to serialize (dataclass, dict, or JSON-serializable).

    Returns:
        str: JSON string representation.

    Example:
        ```python
        from dataclasses import dataclass
        from datetime import datetime

        @dataclass
        class Example:
            name: str
            created: datetime

        obj = Example(name="test", created=datetime.now())
        json_str = _to_json(obj)
        # '{"name": "test", "created": "2024-01-01 12:00:00"}'
        ```
    """
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, default=str, ensure_ascii=False)


def _state_from_json(payload: str) -> StrategyState:
    """Deserialize StrategyState from JSON string.

    Args:
        payload (str): JSON string containing serialized StrategyState.

    Returns:
        StrategyState: Reconstructed strategy state.

    Example:
        ```python
        json_str = '{"strategy_id": "test", "last_ts": null, ...}'
        state = _state_from_json(json_str)
        assert state.strategy_id == "test"
        ```
    """
    data = json.loads(payload)
    return StrategyState(**data)


class DuckDbStrategyStore(StrategyStore):
    """DuckDB implementation of strategy persistence.

    Stores strategy state, positions, trades, and metrics in local DuckDB
    file. Provides efficient storage with SQL query capabilities.

    Schema:
        - strategy_state: Current state snapshots (keyed by strategy_id)
        - positions: Position history (strategy_id, ts, position_id)
        - trades: Trade log (strategy_id, trade_id)
        - metrics: Performance metrics (strategy_id, ts, name)

    Storage format:
        - State: JSON payload with full StrategyState
        - Positions: JSON payload per position
        - Trades: JSON payload per trade
        - Metrics: Normalized table (name-value pairs)

    Attributes:
        path (str): Path to DuckDB file.
        con (duckdb.DuckDBPyConnection): Database connection.

    Example:
        ```python
        from signalflow.data.strategy_store import DuckDbStrategyStore
        from signalflow.core import StrategyState
        from pathlib import Path

        # Create store
        store = DuckDbStrategyStore(str(Path("backtest.duckdb")))
        store.init()

        # Initialize state
        state = StrategyState(strategy_id="my_strategy")
        state.portfolio.cash = 10000.0

        # Save state
        store.save_state(state)

        # Load state
        loaded = store.load_state("my_strategy")
        assert loaded.portfolio.cash == 10000.0

        # Query with SQL
        trades_df = store.con.execute(
            "SELECT * FROM trades WHERE strategy_id = ?",
            ["my_strategy"]
        ).pl()
        ```

    Note:
        JSON serialization handles datetime via default=str.
        All operations use upsert semantics (INSERT ON CONFLICT).
        Connection remains open for query access.

    See Also:
        StrategyStore: Base class with interface definition.
        SCHEMA_SQL: Schema definition for tables.
    """

    def __init__(self, path: str) -> None:
        """Initialize DuckDB store.

        Opens connection to DuckDB file (creates if doesn't exist).

        Args:
            path (str): Path to DuckDB file.

        Example:
            ```python
            store = DuckDbStrategyStore("backtest.duckdb")
            store.init()  # Create tables
            ```
        """
        self.path = path
        self.con = duckdb.connect(path)

    def init(self) -> None:
        """Initialize database schema.

        Creates tables if they don't exist:
            - strategy_state (current snapshots)
            - positions (position history)
            - trades (trade log)
            - metrics (performance metrics)

        Idempotent - safe to call multiple times.

        Example:
            ```python
            store = DuckDbStrategyStore("backtest.duckdb")
            store.init()  # Creates tables
            store.init()  # Safe - no-op if tables exist
            ```

        Note:
            Uses SCHEMA_SQL from schema module.
            Creates indexes for efficient queries.
        """
        self.con.execute(SCHEMA_SQL)

    def load_state(self, strategy_id: str) -> Optional[StrategyState]:
        """Load strategy state from database.

        Retrieves most recent saved state for given strategy.

        Args:
            strategy_id (str): Strategy identifier.

        Returns:
            StrategyState | None: Loaded state or None if not found.

        Example:
            ```python
            # Load existing state
            state = store.load_state("my_strategy")

            if state:
                print(f"Resuming from: {state.last_ts}")
                print(f"Cash: ${state.portfolio.cash}")
            else:
                print("No saved state - starting fresh")
                state = StrategyState(strategy_id="my_strategy")
            ```

        Note:
            Returns None if strategy never saved.
            Deserializes JSON payload to StrategyState.
        """
        row = self.con.execute(
            "SELECT payload_json FROM strategy_state WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()
        if not row:
            return None
        return _state_from_json(row[0])

    def save_state(self, state: StrategyState) -> None:
        """Save strategy state to database.

        Upserts complete state snapshot. Updates if exists, inserts if new.

        Args:
            state (StrategyState): Strategy state to persist.

        Example:
            ```python
            # Update state
            state.last_ts = datetime.now()
            state.portfolio.cash = 9500.0

            # Save (upsert)
            store.save_state(state)

            # Verify
            loaded = store.load_state(state.strategy_id)
            assert loaded.last_ts == state.last_ts
            ```

        Note:
            Uses INSERT ON CONFLICT to handle updates.
            Serializes entire state to JSON.
        """
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
        """Upsert position snapshots to database.

        Records point-in-time position state. Updates if (strategy_id, ts, position_id)
        exists, inserts otherwise.

        Args:
            strategy_id (str): Strategy identifier.
            ts (datetime): Snapshot timestamp.
            positions (Iterable[Position]): Positions to persist.

        Raises:
            ValueError: If position missing id attribute.

        Example:
            ```python
            from datetime import datetime

            # After bar close
            positions = state.portfolio.positions.values()
            store.upsert_positions(
                strategy_id="my_strategy",
                ts=datetime.now(),
                positions=positions
            )

            # Query positions
            positions_df = store.con.execute(
                "SELECT * FROM positions WHERE strategy_id = ?",
                ["my_strategy"]
            ).pl()
            ```

        Note:
            Uses batch executemany for efficiency.
            Silently returns if positions is empty.
            Position id must be present.
        """
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
        """Append trade to log.

        Adds trade to immutable log. Ignores if (strategy_id, trade_id) already exists.

        Args:
            strategy_id (str): Strategy identifier.
            trade (Trade): Trade to persist.

        Raises:
            ValueError: If trade missing id or timestamp.

        Example:
            ```python
            # After trade execution
            trade = Trade(
                position_id="pos_123",
                pair="BTCUSDT",
                side="BUY",
                ts=datetime.now(),
                price=45000.0,
                qty=0.5,
                fee=22.5
            )
            store.append_trade("my_strategy", trade)

            # Query trades
            trades_df = store.con.execute(
                "SELECT * FROM trades WHERE strategy_id = ? ORDER BY ts",
                ["my_strategy"]
            ).pl()
            ```

        Note:
            Uses INSERT ON CONFLICT DO NOTHING for idempotence.
            Accepts both 'id' and 'trade_id' attributes.
            Accepts both 'ts' and 'timestamp' attributes.
        """
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
        """Append metrics snapshot to database.

        Records performance metrics at timestamp. Updates if (strategy_id, ts, name)
        exists, inserts otherwise.

        Args:
            strategy_id (str): Strategy identifier.
            ts (datetime): Metrics timestamp.
            metrics (dict[str, float]): Metric name-value pairs.

        Example:
            ```python
            from datetime import datetime

            # After bar close
            metrics = {
                "total_return": 0.05,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.03,
                "win_rate": 0.65
            }
            store.append_metrics("my_strategy", datetime.now(), metrics)

            # Query metrics time series
            metrics_df = store.con.execute('''
                SELECT ts, name, value FROM metrics
                WHERE strategy_id = ?
                ORDER BY ts, name
            ''', ["my_strategy"]).pl()

            # Pivot for analysis
            pivoted = metrics_df.pivot(
                index="ts",
                columns="name",
                values="value"
            )
            ```

        Note:
            Uses batch executemany for efficiency.
            Silently returns if metrics is empty.
            Values coerced to float.
        """
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
