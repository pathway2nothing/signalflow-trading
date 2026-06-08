# src/signalflow/data/strategy_store/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

from signalflow.core import (
    CashPolicy,
    SfComponentType,
    StrategyState,
    Trade,
    fold,
    portfolios_match,
)


class StrategyStore(ABC):
    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_STORE
    """Abstract base class for strategy state persistence.

    Defines the interface for persisting strategy execution state, including
    portfolio positions, trades, and performance metrics. Stores are responsible
    only for persistence - no business logic.

    Key responsibilities:
        - Initialize storage backend (tables, indexes, etc.)
        - Load/save complete strategy state (snapshot cache) for recovery
        - Append event streams (trades, metrics)
        - Read back the trade log for deterministic replay / snapshot verification

    Common implementations:
        - DuckDB: Local file-based storage
        - PostgreSQL: Shared database for multiple strategies
        - Parquet: Time-series optimized storage

    Persistence patterns:
        - State: Full snapshot CACHE for recovery (load_state, save_state)
        - Events: Append-only log — source of truth (append_trade, read_trades)
        - Metrics: Append-only time series (append_metrics)

    Example:
        ```python
        from signalflow.data.strategy_store import DuckDbStrategyStore
        from pathlib import Path

        # Create store
        store = DuckDbStrategyStore(db_path=Path("backtest.duckdb"))
        store.init()

        try:
            # Load existing state
            state = store.load_state("my_strategy")

            if state is None:
                # Initialize new state
                state = StrategyState(strategy_id="my_strategy")
                state.portfolio.cash = 10000.0

            # Run strategy tick
            # ... execute trades ...

            # Persist trade
            trade = Trade(
                pair="BTCUSDT",
                side="BUY",
                price=45000.0,
                qty=0.5,
                fee=22.5
            )
            store.append_trade("my_strategy", trade)

            # Persist metrics
            metrics = {"total_return": 0.05, "sharpe_ratio": 1.2}
            store.append_metrics("my_strategy", datetime.now(), metrics)

            # Save state checkpoint
            store.save_state(state)

        finally:
            store.close()
        ```

    Note:
        Implementations should be thread-safe for concurrent access.
        State saves should be atomic to prevent corruption on crashes.
        Append operations should be optimized for high-frequency writes.

    See Also:
        StrategyState: The state object being persisted.
        BacktestBroker: Uses store to persist execution events.
    """

    @abstractmethod
    def init(self) -> None:
        """Initialize storage backend.

        Creates necessary tables, indexes, and schema. Idempotent - safe
        to call multiple times. Should handle schema migrations if needed.

        Example:
            ```python
            store = DuckDbStrategyStore(Path("backtest.duckdb"))
            store.init()  # Creates tables
            store.init()  # Safe to call again
            ```

        Note:
            Should create:
                - state table (strategy snapshots)
                - positions table (position history)
                - trades table (trade log)
                - metrics table (performance metrics)
        """
        ...

    @abstractmethod
    def load_state(self, strategy_id: str) -> StrategyState | None:
        """Load strategy state from storage.

        Retrieves most recent saved state for recovery or resumption.
        Returns None if strategy has never been saved.

        Args:
            strategy_id (str): Unique strategy identifier.

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
                print("Starting fresh")
                state = StrategyState(strategy_id="my_strategy")
            ```

        Note:
            Should load most recent checkpoint only.
            Portfolio positions should be reconstructed from state.
        """
        ...

    @abstractmethod
    def save_state(self, state: StrategyState) -> None:
        """Save strategy state to storage.

        Persists complete strategy state as checkpoint. Should be atomic
        to prevent corruption. Overwrites previous state for same strategy_id.

        Args:
            state (StrategyState): Strategy state to persist.

        Example:
            ```python
            # Update state
            state.last_ts = datetime.now()
            state.portfolio.cash = 9500.0

            # Save checkpoint
            store.save_state(state)

            # Can resume from this point later
            resumed_state = store.load_state(state.strategy_id)
            assert resumed_state.last_ts == state.last_ts
            ```

        Note:
            Should be atomic - write to temp then rename/swap.
            Consider compression for large portfolios.
        """
        ...

    @abstractmethod
    def read_trades(self, strategy_id: str) -> list[Trade]:
        """Read the full trade (event) log for a strategy, in chronological order.

        The trade log is the source of truth for portfolio state; this read side
        makes deterministic replay (``core.eventlog.fold``) possible.

        Args:
            strategy_id (str): Strategy identifier.

        Returns:
            list[Trade]: Trades ordered by ``(ts, trade_id)``.
        """
        ...

    def verify_snapshot(
        self,
        strategy_id: str,
        *,
        initial_cash: float = 0.0,
        policy: CashPolicy | None = None,
    ) -> bool:
        """Verify the persisted snapshot equals an event-log replay.

        Folds the trade log into a fresh portfolio and compares it (on the
        trade-derived fields) against the saved snapshot's portfolio. Returns
        True when they agree — i.e. the snapshot is a faithful cache of the log.
        Returns False if no snapshot has been saved.

        Args:
            strategy_id: Strategy identifier.
            initial_cash: Starting cash used when the strategy began.
            policy: Cash-accounting policy used during execution.
        """
        state = self.load_state(strategy_id)
        if state is None:
            return False
        replayed = fold(
            self.read_trades(strategy_id),
            initial_cash=initial_cash,
            policy=policy or CashPolicy(),
        )
        return portfolios_match(replayed, state.portfolio)

    @abstractmethod
    def append_trade(self, strategy_id: str, trade: Trade) -> None:
        """Append trade to log.

        Immutable event log - trades are never updated or deleted.

        Args:
            strategy_id (str): Strategy identifier.
            trade (Trade): Trade to persist.

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

            # Query all trades
            # SELECT * FROM trades WHERE strategy_id = 'my_strategy'
            # ORDER BY ts
            ```

        Note:
            Append-only - optimized for sequential writes.
            Used for trade analysis and PnL verification.
        """
        ...

    @abstractmethod
    def append_metrics(self, strategy_id: str, ts: datetime, metrics: dict[str, float]) -> None:
        """Append metrics snapshot.

        Records performance metrics at specific timestamp. Metrics are
        strategy-defined (returns, sharpe, drawdown, etc.).

        Args:
            strategy_id (str): Strategy identifier.
            ts (datetime): Metrics timestamp.
            metrics (dict[str, float]): Metric name-value pairs.

        Example:
            ```python
            # After each bar
            metrics = {
                "total_return": 0.05,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.03,
                "win_rate": 0.65
            }
            store.append_metrics("my_strategy", datetime.now(), metrics)

            # Query metrics time series
            # SELECT ts, total_return FROM metrics
            # WHERE strategy_id = 'my_strategy'
            # ORDER BY ts
            ```

        Note:
            Append-only time series.
            Metric names should be consistent across snapshots.
            Used for performance visualization and optimization.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close storage connection and cleanup resources.

        Releases database connections, file handles, or other resources.
        Idempotent - safe to call multiple times.
        """
        ...
