"""Result containers for parallel backtest modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from signalflow.core.containers.trade import Trade
    from signalflow.core.containers.position import Position


@dataclass(frozen=True)
class PairResult:
    """Result of backtesting a single pair in isolated mode.

    Attributes:
        pair: Trading pair symbol
        trades: List of executed trades
        final_equity: Final equity value
        final_cash: Final cash balance
        positions: List of all positions (open and closed)
        metrics_history: Per-bar metrics snapshots
        initial_capital: Starting capital for this pair
    """

    pair: str
    trades: list[Trade]
    final_equity: float
    final_cash: float
    positions: list[Position]
    metrics_history: list[dict]
    initial_capital: float

    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return (self.final_equity - self.initial_capital) / self.initial_capital

    @property
    def trade_count(self) -> int:
        """Number of trades executed."""
        return len(self.trades)

    def trades_df(self) -> pl.DataFrame:
        """Convert trades to DataFrame."""
        from signalflow.core.containers.portfolio import Portfolio

        return Portfolio.trades_to_pl(self.trades)

    def metrics_df(self) -> pl.DataFrame:
        """Convert metrics history to DataFrame."""
        if not self.metrics_history:
            return pl.DataFrame()
        return pl.DataFrame(self.metrics_history)


@dataclass
class IsolatedResults:
    """Aggregated results from isolated balance mode.

    Combines results from all pairs into unified metrics.

    Attributes:
        total_equity: Sum of equity across all pairs
        total_return: Weighted average return
        initial_capital: Total starting capital
        pair_results: Per-pair breakdown
        all_trades: Concatenated trades from all pairs
    """

    total_equity: float
    total_return: float
    initial_capital: float
    pair_results: dict[str, PairResult] = field(default_factory=dict)

    @property
    def all_trades(self) -> list[Trade]:
        """Get all trades from all pairs."""
        trades = []
        for result in self.pair_results.values():
            trades.extend(result.trades)
        return trades

    @property
    def total_trades(self) -> int:
        """Total number of trades across all pairs."""
        return sum(r.trade_count for r in self.pair_results.values())

    def trades_df(self) -> pl.DataFrame:
        """All trades as DataFrame."""
        from signalflow.core.containers.portfolio import Portfolio

        return Portfolio.trades_to_pl(self.all_trades)

    def pair_metrics_df(self) -> pl.DataFrame:
        """Per-pair metrics as DataFrame."""
        rows = []
        for pair, result in self.pair_results.items():
            rows.append(
                {
                    "pair": pair,
                    "initial_capital": result.initial_capital,
                    "final_equity": result.final_equity,
                    "total_return": result.total_return,
                    "trade_count": result.trade_count,
                }
            )
        return pl.DataFrame(rows)


@dataclass
class UnlimitedResults:
    """Results from unlimited balance mode.

    Optimized for signal validation without balance constraints.

    Attributes:
        trades_df: All trades with entry/exit details
        total_signals: Number of signals processed
        executed_trades: Number of trades executed
        win_rate: Percentage of winning trades
        avg_return: Average return per trade
        hit_rate: Percentage of signals that hit TP
        avg_bars_in_trade: Average holding period in bars
    """

    trades_df: pl.DataFrame
    total_signals: int
    executed_trades: int
    win_rate: float
    avg_return: float
    hit_rate: float = 0.0
    avg_bars_in_trade: float = 0.0

    @property
    def loss_rate(self) -> float:
        """Percentage of losing trades."""
        return 1.0 - self.win_rate

    def by_pair(self) -> pl.DataFrame:
        """Breakdown by trading pair."""
        if self.trades_df.height == 0:
            return pl.DataFrame()

        return self.trades_df.group_by("pair").agg(
            [
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("pnl").count().alias("trade_count"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
                pl.col("return_pct").mean().alias("avg_return"),
            ]
        )

    def by_side(self) -> pl.DataFrame:
        """Breakdown by trade side (LONG/SHORT)."""
        if self.trades_df.height == 0 or "side" not in self.trades_df.columns:
            return pl.DataFrame()

        return self.trades_df.group_by("side").agg(
            [
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("pnl").count().alias("trade_count"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            ]
        )
