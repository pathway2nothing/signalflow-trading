from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from signalflow.core.containers import Portfolio


@dataclass(slots=True)
class StrategyState:
    """Single source of truth for strategy runtime state.

    Mutable aggregate that tracks all strategy state including portfolio,
    runtime context, metrics, and execution watermarks.

    Lives in `signalflow.core` so both `signalflow.strategy` (logic/execution)
    and `signalflow.data` (persistence) can depend on it without cycles.

    State components:
        - portfolio: Canonical portfolio state (cash + positions)
        - runtime: Flexible bag for cooldowns, watermarks, guards
        - metrics: Latest computed metrics snapshot
        - watermarks: Last processed timestamp and event for idempotency

    Attributes:
        strategy_id (str): Unique strategy identifier.
        last_ts (datetime | None): Last processed timestamp.
        last_event_id (str | None): Last processed event ID (for live idempotency/resume).
        portfolio (Portfolio): Current portfolio state.
        runtime (dict[str, Any]): Runtime context (cooldowns, guards, etc.).
        metrics (dict[str, float]): Latest metrics snapshot.
        metrics_phase_done (set[str]): Phase completion tracking for metrics.

    Example:
        ```python
        from signalflow.core import StrategyState, Portfolio
        from datetime import datetime

        # Initialize state
        state = StrategyState(strategy_id="my_strategy")
        state.portfolio.cash = 10000.0

        # Process bar
        state.touch(ts=datetime(2024, 1, 1, 10, 0))
        state.metrics["total_return"] = 0.05
        state.runtime["last_signal"] = "sma_cross"

        # Save and resume
        saved_state = save_to_db(state)
        resumed_state = load_from_db(strategy_id="my_strategy")

        # Continue from last watermark
        print(f"Resuming from: {resumed_state.last_ts}")

        # Phase-gated metrics
        state.reset_tick_cache()
        # ... compute metrics for new tick ...
        state.metrics_phase_done.add("returns")
        ```

    Note:
        State should be persisted regularly for recovery.
        All portfolio changes flow through fills, not direct modification.
        Use touch() to update watermarks after successful tick commit.
    """

    strategy_id: str

    last_ts: Optional[datetime] = None
    last_event_id: Optional[str] = None

    portfolio: Portfolio = field(default_factory=Portfolio)

    runtime: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    metrics_phase_done: set[str] = field(default_factory=set)

    def touch(self, ts: datetime, event_id: Optional[str] = None) -> None:
        """Update watermarks after successful tick commit.

        Updates last_ts and optionally last_event_id to track processing progress.
        Used for resume/recovery and live idempotency.

        Args:
            ts (datetime): Timestamp of successfully processed tick.
            event_id (str | None): Optional event ID for idempotency tracking.

        Example:
            ```python
            # Process bar successfully
            for bar in bars:
                # ... process trading logic ...

                # Commit and update watermarks
                state.touch(ts=bar.timestamp)
                save_to_db(state)

            # Live trading with event IDs
            for event in event_stream:
                # ... process event ...

                # Track event for idempotency
                state.touch(ts=event.timestamp, event_id=event.id)
                save_to_db(state)
            ```

        Note:
            Call after successful tick processing and before persistence.
            Enables safe resume from last committed state.
        """
        self.last_ts = ts
        if event_id is not None:
            self.last_event_id = event_id

    def reset_tick_cache(self) -> None:
        """Clear phase completion tracking for new tick.

        Resets metrics_phase_done set at the start of each tick
        for phase-gated metrics computation.

        Use when computing metrics incrementally across phases
        (e.g., pre-trade, post-trade, end-of-bar).

        Example:
            ```python
            # At start of each tick
            state.reset_tick_cache()

            # Phase 1: Pre-trade metrics
            if "returns" not in state.metrics_phase_done:
                compute_return_metrics(state)
                state.metrics_phase_done.add("returns")

            # Phase 2: Post-trade metrics
            if "drawdown" not in state.metrics_phase_done:
                compute_drawdown_metrics(state)
                state.metrics_phase_done.add("drawdown")

            # Next tick - reset and recompute
            state.reset_tick_cache()
            ```

        Note:
            Only needed if using phase-gated metrics pattern.
            Otherwise, metrics_phase_done can be ignored.
        """
        self.metrics_phase_done.clear()
