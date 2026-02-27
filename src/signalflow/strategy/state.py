"""State persistence for live trading.

This module implements state persistence from SFFLOW.md specification:
- Position state (critical)
- Risk state (critical)
- Signal state (important)
- Execution state (important)

Backends:
- Redis: Fast, optional persistence (RDB/AOF), multi-process
- DuckDB: Always persistent, embedded, single bot

Example:
    >>> from signalflow.strategy.state import StateManager, StateBackend
    >>> state = StateManager.from_config({
    ...     "backend": "redis",
    ...     "redis": {"url": "redis://localhost:6379"},
    ... })
    >>> await state.save_position(position)
    >>> positions = await state.get_positions()
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore


class StateBackend(StrEnum):
    """State persistence backends."""

    REDIS = "redis"
    DUCKDB = "duckdb"
    MEMORY = "memory"  # For testing


class RecoveryMode(StrEnum):
    """Recovery modes after restart."""

    SYNC = "sync"  # Sync state with exchange
    RESTORE = "restore"  # Restore from persistence only
    CLOSE_ALL = "close_all"  # Close all positions
    MANUAL = "manual"  # Require manual intervention


class OrphanPositionAction(StrEnum):
    """Action for orphan positions found on exchange."""

    CLOSE = "close"  # Close the position
    ADOPT = "adopt"  # Adopt into state
    MANUAL = "manual"  # Manual intervention


@dataclass
class Position:
    """Open position state.

    Attributes:
        id: Unique position ID
        pair: Trading pair
        side: Position side (long/short)
        size: Position size
        entry_price: Entry price
        entry_ts: Entry timestamp
        sl: Stop loss price (optional)
        tp: Take profit price (optional)
        metadata: Additional metadata
    """

    id: str
    pair: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_ts: datetime
    sl: float | None = None
    tp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        data["entry_ts"] = self.entry_ts.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        """Create from dict."""
        entry_ts = data.get("entry_ts")
        if isinstance(entry_ts, str):
            entry_ts = datetime.fromisoformat(entry_ts)
        return cls(
            id=data["id"],
            pair=data["pair"],
            side=data["side"],
            size=data["size"],
            entry_price=data["entry_price"],
            entry_ts=entry_ts,
            sl=data.get("sl"),
            tp=data.get("tp"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PendingOrder:
    """Pending order state.

    Attributes:
        id: Order ID
        pair: Trading pair
        side: Order side
        order_type: Order type (limit, market, etc.)
        price: Order price
        size: Order size
        status: Order status
        created_at: Creation timestamp
    """

    id: str
    pair: str
    side: str
    order_type: str
    price: float | None
    size: float
    status: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingOrder:
        """Create from dict."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            id=data["id"],
            pair=data["pair"],
            side=data["side"],
            order_type=data["order_type"],
            price=data.get("price"),
            size=data["size"],
            status=data["status"],
            created_at=created_at,
        )


@dataclass
class RiskState:
    """Risk tracking state.

    Attributes:
        daily_pnl: Today's PnL
        daily_trades: Today's trade count
        consecutive_losses: Consecutive losing trades
        current_drawdown: Current drawdown from peak
        peak_equity: Peak equity value
        circuit_breaker_active: Whether circuit breaker is triggered
        circuit_breaker_reason: Reason for circuit breaker
        circuit_breaker_until: When circuit breaker expires
    """

    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str | None = None
    circuit_breaker_until: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        data = asdict(self)
        if self.circuit_breaker_until:
            data["circuit_breaker_until"] = self.circuit_breaker_until.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskState:
        """Create from dict."""
        cb_until = data.get("circuit_breaker_until")
        if isinstance(cb_until, str):
            cb_until = datetime.fromisoformat(cb_until)
        return cls(
            daily_pnl=data.get("daily_pnl", 0.0),
            daily_trades=data.get("daily_trades", 0),
            consecutive_losses=data.get("consecutive_losses", 0),
            current_drawdown=data.get("current_drawdown", 0.0),
            peak_equity=data.get("peak_equity", 0.0),
            circuit_breaker_active=data.get("circuit_breaker_active", False),
            circuit_breaker_reason=data.get("circuit_breaker_reason"),
            circuit_breaker_until=cb_until,
        )


@dataclass
class SignalState:
    """Signal tracking state.

    Attributes:
        last_processed_ts: Last processed signal timestamp
        last_processed_pair: Last processed pair
        last_processed_id: Last processed signal ID
        cooldowns: Pair cooldown expiry times
        recent_signal_ids: Recent signal IDs (for deduplication)
    """

    last_processed_ts: datetime | None = None
    last_processed_pair: str | None = None
    last_processed_id: str | None = None
    cooldowns: dict[str, datetime] = field(default_factory=dict)
    recent_signal_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "last_processed_ts": self.last_processed_ts.isoformat() if self.last_processed_ts else None,
            "last_processed_pair": self.last_processed_pair,
            "last_processed_id": self.last_processed_id,
            "cooldowns": {k: v.isoformat() for k, v in self.cooldowns.items()},
            "recent_signal_ids": self.recent_signal_ids[-100:],  # Keep last 100
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalState:
        """Create from dict."""
        last_ts = data.get("last_processed_ts")
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts)

        cooldowns = {}
        for k, v in data.get("cooldowns", {}).items():
            if isinstance(v, str):
                cooldowns[k] = datetime.fromisoformat(v)

        return cls(
            last_processed_ts=last_ts,
            last_processed_pair=data.get("last_processed_pair"),
            last_processed_id=data.get("last_processed_id"),
            cooldowns=cooldowns,
            recent_signal_ids=data.get("recent_signal_ids", []),
        )


class BaseStateBackend(ABC):
    """Base class for state backends."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to backend."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connection."""
        ...

    @abstractmethod
    async def save_positions(self, positions: list[Position]) -> None:
        """Save open positions."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get open positions."""
        ...

    @abstractmethod
    async def save_pending_orders(self, orders: list[PendingOrder]) -> None:
        """Save pending orders."""
        ...

    @abstractmethod
    async def get_pending_orders(self) -> list[PendingOrder]:
        """Get pending orders."""
        ...

    @abstractmethod
    async def save_risk_state(self, state: RiskState) -> None:
        """Save risk state."""
        ...

    @abstractmethod
    async def get_risk_state(self) -> RiskState:
        """Get risk state."""
        ...

    @abstractmethod
    async def save_signal_state(self, state: SignalState) -> None:
        """Save signal state."""
        ...

    @abstractmethod
    async def get_signal_state(self) -> SignalState:
        """Get signal state."""
        ...

    @abstractmethod
    async def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        ...

    @abstractmethod
    async def get_last_heartbeat(self) -> datetime | None:
        """Get last heartbeat timestamp."""
        ...


class MemoryStateBackend(BaseStateBackend):
    """In-memory state backend for testing."""

    def __init__(self) -> None:
        """Initialize memory backend."""
        self._positions: list[Position] = []
        self._orders: list[PendingOrder] = []
        self._risk: RiskState = RiskState()
        self._signal: SignalState = SignalState()
        self._heartbeat: datetime | None = None

    async def connect(self) -> None:
        """No-op for memory backend."""
        pass

    async def close(self) -> None:
        """No-op for memory backend."""
        pass

    async def save_positions(self, positions: list[Position]) -> None:
        """Save positions in memory."""
        self._positions = positions.copy()

    async def get_positions(self) -> list[Position]:
        """Get positions from memory."""
        return self._positions.copy()

    async def save_pending_orders(self, orders: list[PendingOrder]) -> None:
        """Save orders in memory."""
        self._orders = orders.copy()

    async def get_pending_orders(self) -> list[PendingOrder]:
        """Get orders from memory."""
        return self._orders.copy()

    async def save_risk_state(self, state: RiskState) -> None:
        """Save risk state in memory."""
        self._risk = state

    async def get_risk_state(self) -> RiskState:
        """Get risk state from memory."""
        return self._risk

    async def save_signal_state(self, state: SignalState) -> None:
        """Save signal state in memory."""
        self._signal = state

    async def get_signal_state(self) -> SignalState:
        """Get signal state from memory."""
        return self._signal

    async def update_heartbeat(self) -> None:
        """Update heartbeat."""
        self._heartbeat = datetime.utcnow()

    async def get_last_heartbeat(self) -> datetime | None:
        """Get last heartbeat."""
        return self._heartbeat


class RedisStateBackend(BaseStateBackend):
    """Redis state backend.

    Key schema:
        sf:{flow_id}:positions:open -> Hash {pos_id: json}
        sf:{flow_id}:positions:pending -> Hash {order_id: json}
        sf:{flow_id}:risk:daily -> Hash {pnl, trades, losses, ...}
        sf:{flow_id}:risk:circuit_breaker -> Hash {active, reason, until}
        sf:{flow_id}:signals:processed -> Sorted Set {signal_id: timestamp}
        sf:{flow_id}:signals:cooldowns -> Hash {pair: next_allowed_ts}
        sf:{flow_id}:execution:heartbeat -> String (timestamp)
    """

    def __init__(
        self,
        flow_id: str,
        url: str = "redis://localhost:6379",
        key_prefix: str = "sf",
    ):
        """Initialize Redis backend.

        Args:
            flow_id: Flow identifier
            url: Redis URL
            key_prefix: Key prefix
        """
        if aioredis is None:
            raise ImportError("redis package required for Redis backend")

        self.flow_id = flow_id
        self.url = url
        self.key_prefix = key_prefix
        self._client: aioredis.Redis | None = None

    def _key(self, *parts: str) -> str:
        """Build Redis key."""
        return f"{self.key_prefix}:{self.flow_id}:{':'.join(parts)}"

    async def connect(self) -> None:
        """Connect to Redis."""
        self._client = await aioredis.from_url(self.url)
        logger.info(f"Connected to Redis at {self.url}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def save_positions(self, positions: list[Position]) -> None:
        """Save positions to Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("positions", "open")

        # Clear and repopulate
        await self._client.delete(key)
        if positions:
            mapping = {p.id: json.dumps(p.to_dict()) for p in positions}
            await self._client.hset(key, mapping=mapping)

    async def get_positions(self) -> list[Position]:
        """Get positions from Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("positions", "open")
        data = await self._client.hgetall(key)

        positions = []
        for _pos_id, pos_json in data.items():
            pos_data = json.loads(pos_json)
            positions.append(Position.from_dict(pos_data))

        return positions

    async def save_pending_orders(self, orders: list[PendingOrder]) -> None:
        """Save pending orders to Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("positions", "pending")

        await self._client.delete(key)
        if orders:
            mapping = {o.id: json.dumps(o.to_dict()) for o in orders}
            await self._client.hset(key, mapping=mapping)

    async def get_pending_orders(self) -> list[PendingOrder]:
        """Get pending orders from Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("positions", "pending")
        data = await self._client.hgetall(key)

        orders = []
        for _order_id, order_json in data.items():
            order_data = json.loads(order_json)
            orders.append(PendingOrder.from_dict(order_data))

        return orders

    async def save_risk_state(self, state: RiskState) -> None:
        """Save risk state to Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("risk", "daily")
        await self._client.hset(key, mapping={
            "daily_pnl": str(state.daily_pnl),
            "daily_trades": str(state.daily_trades),
            "consecutive_losses": str(state.consecutive_losses),
            "current_drawdown": str(state.current_drawdown),
            "peak_equity": str(state.peak_equity),
        })

        cb_key = self._key("risk", "circuit_breaker")
        await self._client.hset(cb_key, mapping={
            "active": "1" if state.circuit_breaker_active else "0",
            "reason": state.circuit_breaker_reason or "",
            "until": state.circuit_breaker_until.isoformat() if state.circuit_breaker_until else "",
        })

    async def get_risk_state(self) -> RiskState:
        """Get risk state from Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("risk", "daily")
        data = await self._client.hgetall(key)

        cb_key = self._key("risk", "circuit_breaker")
        cb_data = await self._client.hgetall(cb_key)

        cb_until = None
        if cb_data.get(b"until") or cb_data.get("until"):
            until_str = cb_data.get(b"until", cb_data.get("until", b""))
            if isinstance(until_str, bytes):
                until_str = until_str.decode()
            if until_str:
                cb_until = datetime.fromisoformat(until_str)

        def get_val(d: dict, k: str, default: str = "0") -> str:
            v = d.get(k.encode(), d.get(k, default))
            return v.decode() if isinstance(v, bytes) else str(v)

        return RiskState(
            daily_pnl=float(get_val(data, "daily_pnl", "0")),
            daily_trades=int(get_val(data, "daily_trades", "0")),
            consecutive_losses=int(get_val(data, "consecutive_losses", "0")),
            current_drawdown=float(get_val(data, "current_drawdown", "0")),
            peak_equity=float(get_val(data, "peak_equity", "0")),
            circuit_breaker_active=get_val(cb_data, "active", "0") == "1",
            circuit_breaker_reason=get_val(cb_data, "reason", "") or None,
            circuit_breaker_until=cb_until,
        )

    async def save_signal_state(self, state: SignalState) -> None:
        """Save signal state to Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        # Save last processed
        key = self._key("signals", "last")
        await self._client.hset(key, mapping={
            "ts": state.last_processed_ts.isoformat() if state.last_processed_ts else "",
            "pair": state.last_processed_pair or "",
            "id": state.last_processed_id or "",
        })

        # Save cooldowns
        cd_key = self._key("signals", "cooldowns")
        await self._client.delete(cd_key)
        if state.cooldowns:
            mapping = {k: v.isoformat() for k, v in state.cooldowns.items()}
            await self._client.hset(cd_key, mapping=mapping)

        # Save recent signal IDs (sorted set with timestamp scores)
        proc_key = self._key("signals", "processed")
        now = datetime.utcnow().timestamp()
        for sig_id in state.recent_signal_ids[-100:]:
            await self._client.zadd(proc_key, {sig_id: now})

        # Trim old entries (keep last 24h)
        cutoff = now - 86400
        await self._client.zremrangebyscore(proc_key, 0, cutoff)

    async def get_signal_state(self) -> SignalState:
        """Get signal state from Redis."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("signals", "last")
        data = await self._client.hgetall(key)

        def get_str(d: dict, k: str) -> str:
            v = d.get(k.encode(), d.get(k, b""))
            return v.decode() if isinstance(v, bytes) else str(v)

        last_ts = None
        ts_str = get_str(data, "ts")
        if ts_str:
            last_ts = datetime.fromisoformat(ts_str)

        # Get cooldowns
        cd_key = self._key("signals", "cooldowns")
        cd_data = await self._client.hgetall(cd_key)
        cooldowns = {}
        for k, v in cd_data.items():
            pair = k.decode() if isinstance(k, bytes) else k
            ts = v.decode() if isinstance(v, bytes) else v
            if ts:
                cooldowns[pair] = datetime.fromisoformat(ts)

        # Get recent signal IDs
        proc_key = self._key("signals", "processed")
        recent_ids = await self._client.zrange(proc_key, -100, -1)
        recent = [s.decode() if isinstance(s, bytes) else s for s in recent_ids]

        return SignalState(
            last_processed_ts=last_ts,
            last_processed_pair=get_str(data, "pair") or None,
            last_processed_id=get_str(data, "id") or None,
            cooldowns=cooldowns,
            recent_signal_ids=recent,
        )

    async def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("execution", "heartbeat")
        await self._client.set(key, datetime.utcnow().isoformat())

    async def get_last_heartbeat(self) -> datetime | None:
        """Get last heartbeat timestamp."""
        if not self._client:
            raise RuntimeError("Not connected")

        key = self._key("execution", "heartbeat")
        data = await self._client.get(key)

        if data:
            ts_str = data.decode() if isinstance(data, bytes) else data
            return datetime.fromisoformat(ts_str)

        return None


class DuckDBStateBackend(BaseStateBackend):
    """DuckDB state backend.

    Uses embedded DuckDB for persistent state storage.
    Better for single-bot scenarios with simpler setup.
    """

    def __init__(
        self,
        flow_id: str,
        path: str | Path = "state/{flow_id}.db",
    ):
        """Initialize DuckDB backend.

        Args:
            flow_id: Flow identifier
            path: Database path (supports {flow_id} placeholder)
        """
        if duckdb is None:
            raise ImportError("duckdb package required for DuckDB backend")

        self.flow_id = flow_id
        self.path = Path(str(path).format(flow_id=flow_id))
        self._conn: duckdb.DuckDBPyConnection | None = None

    async def connect(self) -> None:
        """Connect to DuckDB."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._init_schema()
        logger.info(f"Connected to DuckDB at {self.path}")

    def _init_schema(self) -> None:
        """Initialize database schema."""
        if not self._conn:
            return

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id VARCHAR PRIMARY KEY,
                data JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_orders (
                id VARCHAR PRIMARY KEY,
                data JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                id VARCHAR PRIMARY KEY DEFAULT 'current',
                data JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_state (
                id VARCHAR PRIMARY KEY DEFAULT 'current',
                data JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS heartbeat (
                id VARCHAR PRIMARY KEY DEFAULT 'current',
                ts TIMESTAMP NOT NULL
            )
        """)

    async def close(self) -> None:
        """Close DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def save_positions(self, positions: list[Position]) -> None:
        """Save positions to DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute("DELETE FROM positions")
        for pos in positions:
            self._conn.execute(
                "INSERT INTO positions (id, data) VALUES (?, ?)",
                [pos.id, json.dumps(pos.to_dict())]
            )

    async def get_positions(self) -> list[Position]:
        """Get positions from DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        result = self._conn.execute("SELECT data FROM positions").fetchall()
        return [Position.from_dict(json.loads(row[0])) for row in result]

    async def save_pending_orders(self, orders: list[PendingOrder]) -> None:
        """Save pending orders to DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute("DELETE FROM pending_orders")
        for order in orders:
            self._conn.execute(
                "INSERT INTO pending_orders (id, data) VALUES (?, ?)",
                [order.id, json.dumps(order.to_dict())]
            )

    async def get_pending_orders(self) -> list[PendingOrder]:
        """Get pending orders from DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        result = self._conn.execute("SELECT data FROM pending_orders").fetchall()
        return [PendingOrder.from_dict(json.loads(row[0])) for row in result]

    async def save_risk_state(self, state: RiskState) -> None:
        """Save risk state to DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO risk_state (id, data, updated_at)
            VALUES ('current', ?, CURRENT_TIMESTAMP)
            """,
            [json.dumps(state.to_dict())]
        )

    async def get_risk_state(self) -> RiskState:
        """Get risk state from DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        result = self._conn.execute(
            "SELECT data FROM risk_state WHERE id = 'current'"
        ).fetchone()

        if result:
            return RiskState.from_dict(json.loads(result[0]))
        return RiskState()

    async def save_signal_state(self, state: SignalState) -> None:
        """Save signal state to DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO signal_state (id, data, updated_at)
            VALUES ('current', ?, CURRENT_TIMESTAMP)
            """,
            [json.dumps(state.to_dict())]
        )

    async def get_signal_state(self) -> SignalState:
        """Get signal state from DuckDB."""
        if not self._conn:
            raise RuntimeError("Not connected")

        result = self._conn.execute(
            "SELECT data FROM signal_state WHERE id = 'current'"
        ).fetchone()

        if result:
            return SignalState.from_dict(json.loads(result[0]))
        return SignalState()

    async def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO heartbeat (id, ts)
            VALUES ('current', CURRENT_TIMESTAMP)
            """
        )

    async def get_last_heartbeat(self) -> datetime | None:
        """Get last heartbeat timestamp."""
        if not self._conn:
            raise RuntimeError("Not connected")

        result = self._conn.execute(
            "SELECT ts FROM heartbeat WHERE id = 'current'"
        ).fetchone()

        if result:
            return result[0]
        return None


@dataclass
class StateConfig:
    """State persistence configuration.

    Attributes:
        backend: Backend type (redis, duckdb, memory)
        redis: Redis configuration
        duckdb: DuckDB configuration
        persist: What to persist
        recovery: Recovery configuration
    """

    backend: StateBackend = StateBackend.MEMORY
    redis: dict[str, Any] = field(default_factory=dict)
    duckdb: dict[str, Any] = field(default_factory=dict)
    persist: list[str] = field(default_factory=lambda: [
        "positions", "pending_orders", "daily_stats",
        "circuit_breaker", "signal_cooldowns"
    ])
    recovery: dict[str, Any] = field(default_factory=lambda: {
        "mode": "sync",
        "orphan_positions": "close",
        "max_state_age": "24h",
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateConfig:
        """Create from dict."""
        backend_str = data.get("backend", "memory")
        return cls(
            backend=StateBackend(backend_str),
            redis=data.get("redis", {}),
            duckdb=data.get("duckdb", {}),
            persist=data.get("persist", cls.__dataclass_fields__["persist"].default_factory()),
            recovery=data.get("recovery", {}),
        )


class StateManager:
    """High-level state manager.

    Coordinates state persistence and recovery.

    Example:
        >>> manager = StateManager.from_config(flow_id="my_flow", config={...})
        >>> await manager.connect()
        >>> await manager.save_position(position)
        >>> positions = await manager.get_positions()
    """

    def __init__(
        self,
        flow_id: str,
        backend: BaseStateBackend,
        config: StateConfig,
    ):
        """Initialize state manager.

        Args:
            flow_id: Flow identifier
            backend: State backend
            config: State configuration
        """
        self.flow_id = flow_id
        self.backend = backend
        self.config = config

    @classmethod
    def from_config(cls, flow_id: str, config: dict[str, Any] | StateConfig) -> StateManager:
        """Create from configuration.

        Args:
            flow_id: Flow identifier
            config: State configuration

        Returns:
            Configured StateManager
        """
        if isinstance(config, dict):
            config = StateConfig.from_dict(config)

        # Create backend
        if config.backend == StateBackend.REDIS:
            backend = RedisStateBackend(
                flow_id=flow_id,
                url=config.redis.get("url", "redis://localhost:6379"),
                key_prefix=config.redis.get("key_prefix", "sf"),
            )
        elif config.backend == StateBackend.DUCKDB:
            backend = DuckDBStateBackend(
                flow_id=flow_id,
                path=config.duckdb.get("path", "state/{flow_id}.db"),
            )
        else:
            backend = MemoryStateBackend()

        return cls(flow_id=flow_id, backend=backend, config=config)

    async def connect(self) -> None:
        """Connect to backend."""
        await self.backend.connect()

    async def close(self) -> None:
        """Close backend connection."""
        await self.backend.close()

    async def __aenter__(self) -> StateManager:
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Context manager exit."""
        await self.close()

    # Position management
    async def save_position(self, position: Position) -> None:
        """Save a single position."""
        positions = await self.backend.get_positions()
        positions = [p for p in positions if p.id != position.id]
        positions.append(position)
        await self.backend.save_positions(positions)

    async def remove_position(self, position_id: str) -> None:
        """Remove a position."""
        positions = await self.backend.get_positions()
        positions = [p for p in positions if p.id != position_id]
        await self.backend.save_positions(positions)

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        return await self.backend.get_positions()

    # Order management
    async def save_order(self, order: PendingOrder) -> None:
        """Save a pending order."""
        orders = await self.backend.get_pending_orders()
        orders = [o for o in orders if o.id != order.id]
        orders.append(order)
        await self.backend.save_pending_orders(orders)

    async def remove_order(self, order_id: str) -> None:
        """Remove a pending order."""
        orders = await self.backend.get_pending_orders()
        orders = [o for o in orders if o.id != order_id]
        await self.backend.save_pending_orders(orders)

    async def get_pending_orders(self) -> list[PendingOrder]:
        """Get pending orders."""
        return await self.backend.get_pending_orders()

    # Risk state
    async def get_risk_state(self) -> RiskState:
        """Get risk state."""
        return await self.backend.get_risk_state()

    async def save_risk_state(self, state: RiskState) -> None:
        """Save risk state."""
        await self.backend.save_risk_state(state)

    async def update_daily_pnl(self, pnl_change: float) -> RiskState:
        """Update daily PnL."""
        state = await self.backend.get_risk_state()
        state.daily_pnl += pnl_change
        state.daily_trades += 1
        await self.backend.save_risk_state(state)
        return state

    async def trigger_circuit_breaker(
        self,
        reason: str,
        duration: timedelta = timedelta(hours=1),
    ) -> None:
        """Trigger circuit breaker."""
        state = await self.backend.get_risk_state()
        state.circuit_breaker_active = True
        state.circuit_breaker_reason = reason
        state.circuit_breaker_until = datetime.utcnow() + duration
        await self.backend.save_risk_state(state)
        logger.warning(f"Circuit breaker triggered: {reason}")

    async def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is active."""
        state = await self.backend.get_risk_state()

        if not state.circuit_breaker_active:
            return False

        # Check if expired
        if state.circuit_breaker_until and datetime.utcnow() > state.circuit_breaker_until:
            state.circuit_breaker_active = False
            state.circuit_breaker_reason = None
            state.circuit_breaker_until = None
            await self.backend.save_risk_state(state)
            logger.info("Circuit breaker expired")
            return False

        return True

    # Signal state
    async def get_signal_state(self) -> SignalState:
        """Get signal state."""
        return await self.backend.get_signal_state()

    async def mark_signal_processed(
        self,
        signal_id: str,
        pair: str,
        timestamp: datetime,
    ) -> None:
        """Mark signal as processed."""
        state = await self.backend.get_signal_state()
        state.last_processed_id = signal_id
        state.last_processed_pair = pair
        state.last_processed_ts = timestamp
        state.recent_signal_ids.append(signal_id)
        state.recent_signal_ids = state.recent_signal_ids[-100:]
        await self.backend.save_signal_state(state)

    async def is_signal_processed(self, signal_id: str) -> bool:
        """Check if signal was already processed."""
        state = await self.backend.get_signal_state()
        return signal_id in state.recent_signal_ids

    async def set_cooldown(self, pair: str, duration: timedelta) -> None:
        """Set cooldown for pair."""
        state = await self.backend.get_signal_state()
        state.cooldowns[pair] = datetime.utcnow() + duration
        await self.backend.save_signal_state(state)

    async def is_on_cooldown(self, pair: str) -> bool:
        """Check if pair is on cooldown."""
        state = await self.backend.get_signal_state()
        cooldown = state.cooldowns.get(pair)
        return bool(cooldown and datetime.utcnow() < cooldown)

    # Heartbeat
    async def heartbeat(self) -> None:
        """Update heartbeat."""
        await self.backend.update_heartbeat()

    async def get_last_heartbeat(self) -> datetime | None:
        """Get last heartbeat."""
        return await self.backend.get_last_heartbeat()

    async def check_stale(self, max_age: timedelta = timedelta(hours=24)) -> bool:
        """Check if state is stale (no recent heartbeat)."""
        last = await self.backend.get_last_heartbeat()
        if last is None:
            return True
        return datetime.utcnow() - last > max_age
