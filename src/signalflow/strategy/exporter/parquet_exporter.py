"""Parquet exporter for backtest training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from signalflow.core import Position, Signals, StrategyState


@dataclass
class BacktestExporter:
    """Export backtest results for external ML training.

    Exports per-bar data (signals + metrics) and per-trade data to Parquet format.
    Does NOT include raw OHLCV prices - only signals and derived metrics.

    Output Files:
        - {output_path}/bars.parquet: Per-bar signals and metrics
        - {output_path}/trades.parquet: Entry/exit trade pairs with outcomes

    Attributes:
        pair_col: Column name for pair in signals.
        ts_col: Column name for timestamp in signals.

    Example:
        >>> exporter = BacktestExporter()
        >>>
        >>> # During backtest (can be integrated with runner)
        >>> for ts in timestamps:
        ...     # ... process bar ...
        ...     exporter.export_bar(ts, signals, state.metrics, state)
        ...
        ...     for trade in bar_trades:
        ...         exporter.export_trade(trade_data)
        >>>
        >>> # After backtest
        >>> exporter.finalize(Path("./training_data"))
        >>>
        >>> # Load for training
        >>> bars = pl.read_parquet("./training_data/bars.parquet")
        >>> trades = pl.read_parquet("./training_data/trades.parquet")
    """

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    _bar_records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _trade_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def export_bar(
        self,
        timestamp: datetime,
        signals: Signals,
        metrics: dict[str, float],
        state: StrategyState,
    ) -> None:
        """Record bar data for export.

        Records:
            - Timestamp
            - All signals for this bar (flattened)
            - All metrics values
            - Position summary (count, total exposure)

        Args:
            timestamp: Current bar timestamp.
            signals: Current bar signals.
            metrics: Current strategy metrics.
            state: Current strategy state.
        """
        if signals is None or signals.value.height == 0:
            # Still record metrics even without signals
            record = {
                self.ts_col: timestamp,
                **{f"metric_{k}": v for k, v in metrics.items() if k != "timestamp"},
                "open_position_count": len(state.portfolio.open_positions()),
            }
            self._bar_records.append(record)
            return

        # For each signal, create a record with metrics
        for row in signals.value.iter_rows(named=True):
            record = {
                self.ts_col: timestamp,
                self.pair_col: row.get(self.pair_col, ""),
                "signal_type": row.get("signal_type", ""),
                "signal": row.get("signal", 0),
                "probability": row.get("probability", 0.0),
                **{f"metric_{k}": v for k, v in metrics.items() if k != "timestamp"},
                "open_position_count": len(state.portfolio.open_positions()),
            }
            self._bar_records.append(record)

    def export_trade(
        self,
        trade_data: dict[str, Any],
    ) -> None:
        """Record completed trade for export.

        Args:
            trade_data: Dictionary containing trade information.
                Expected keys:
                - position_id
                - pair
                - entry_time, entry_price
                - exit_time, exit_price (if closed)
                - realized_pnl
                - hold_duration_bars
                - entry_signal_type, entry_confidence
                - exit_reason
        """
        self._trade_records.append(trade_data)

    def export_position_close(
        self,
        position: Position,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str = "unknown",
    ) -> None:
        """Convenience method to export when a position closes.

        Args:
            position: The position being closed.
            exit_time: Time of exit.
            exit_price: Price at exit.
            exit_reason: Reason for exit (e.g., "take_profit", "stop_loss", "model_exit").
        """
        entry_meta = position.meta or {}

        trade_data = {
            "position_id": position.id,
            "pair": position.pair,
            "position_type": position.position_type.value,
            "entry_time": position.entry_time,
            "entry_price": position.entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "qty": position.qty,
            "realized_pnl": position.realized_pnl,
            "total_pnl": position.total_pnl,
            "fees_paid": position.fees_paid,
            "signal_strength": position.signal_strength,
            "exit_reason": exit_reason,
            "entry_signal_type": entry_meta.get("signal_type", ""),
            "model_confidence": entry_meta.get("model_confidence", 0.0),
        }
        self._trade_records.append(trade_data)

    def finalize(self, output_path: Path) -> None:
        """Write all data to Parquet files.

        Args:
            output_path: Directory to write output files.
                Creates directory if it doesn't exist.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export bars
        if self._bar_records:
            bars_df = pl.DataFrame(self._bar_records)
            bars_df.write_parquet(output_path / "bars.parquet")

        # Export trades
        if self._trade_records:
            trades_df = pl.DataFrame(self._trade_records)
            trades_df.write_parquet(output_path / "trades.parquet")

    def reset(self) -> None:
        """Clear all recorded data."""
        self._bar_records.clear()
        self._trade_records.clear()

    @property
    def bar_count(self) -> int:
        """Number of bar records collected."""
        return len(self._bar_records)

    @property
    def trade_count(self) -> int:
        """Number of trade records collected."""
        return len(self._trade_records)
