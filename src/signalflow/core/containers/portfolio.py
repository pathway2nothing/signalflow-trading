from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal

import polars as pl

from signalflow.core.containers.position import Position
from signalflow.core.containers.trade import Trade


@dataclass(slots=True)
class Portfolio:
    """Portfolio snapshot (pure domain).

    Tracks cash and open/closed positions. Provides equity calculation
    and DataFrame conversion utilities.

    Portfolio state should only be modified through broker operations
    to maintain accounting consistency.

    Attributes:
        cash (float): Available cash balance.
        positions (dict[str, Position]): Dictionary of positions keyed by position ID.

    Example:
        ```python
        from signalflow.core import Portfolio, Position, PositionType

        # Initialize portfolio
        portfolio = Portfolio(cash=10000.0)

        # Add position
        position = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=45000.0,
            qty=0.5
        )
        portfolio.positions[position.id] = position

        # Calculate equity
        prices = {"BTCUSDT": 46000.0, "ETHUSDT": 3200.0}
        equity = portfolio.equity(prices=prices)

        # Get open positions
        open_positions = portfolio.open_positions()

        # Convert to DataFrame
        positions_df = Portfolio.positions_to_pl(open_positions)
        ```

    Note:
        Portfolio state should only be modified through broker operations.
        Direct manipulation may lead to accounting inconsistencies.
    """

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def open_positions(self) -> list[Position]:
        """Get list of open (non-closed) positions.

        Returns:
            list[Position]: Open positions.

        Example:
            ```python
            open = portfolio.open_positions()
            print(f"Open positions: {len(open)}")

            for pos in open:
                print(f"{pos.pair}: {pos.qty} @ ${pos.entry_price}")
            ```
        """
        return [p for p in self.positions.values() if not p.is_closed]

    def equity(self, *, prices: dict[str, float]) -> float:
        """Calculate total portfolio equity.

        Equity = cash + sum(side_sign * price * qty for all positions)

        Executor must keep accounting consistent for accurate equity.

        Args:
            prices (dict[str, float]): Current prices per pair.

        Returns:
            float: Total equity in currency units.

        Example:
            ```python
            prices = {
                "BTCUSDT": 46000.0,
                "ETHUSDT": 3200.0
            }
            equity = portfolio.equity(prices=prices)
            print(f"Total equity: ${equity:,.2f}")

            # Track equity over time
            equity_history = []
            for ts, prices in price_history:
                eq = portfolio.equity(prices=prices)
                equity_history.append((ts, eq))
            ```

        Note:
            If price not in dict, uses position's last_price.
            Executor must keep accounting consistent.
        """
        eq = self.cash
        for p in self.positions.values():
            px = prices.get(p.pair, p.last_price)
            eq += p.side_sign * px * p.qty
        return eq

    @staticmethod
    def positions_to_pl(positions: Iterable[Position]) -> pl.DataFrame:
        """Convert positions to Polars DataFrame.

        Args:
            positions (Iterable[Position]): Positions to convert.

        Returns:
            pl.DataFrame: DataFrame with position data. Empty if no positions.

        Example:
            ```python
            # Convert all positions
            all_df = Portfolio.positions_to_pl(portfolio.positions.values())

            # Convert only open positions
            open_df = Portfolio.positions_to_pl(portfolio.open_positions())

            # Analyze positions
            print(open_df.select(["pair", "qty", "realized_pnl"]))

            # Group by pair
            by_pair = open_df.group_by("pair").agg([
                pl.col("qty").sum().alias("total_qty"),
                pl.col("realized_pnl").sum().alias("total_pnl")
            ])
            ```
        """
        if not positions:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "id": p.id,
                    "is_closed": p.is_closed,
                    "pair": p.pair,
                    "position_type": p.position_type.value,
                    "signal_strength": p.signal_strength,
                    "entry_time": p.entry_time,
                    "last_time": p.last_time,
                    "entry_price": p.entry_price,
                    "last_price": p.last_price,
                    "qty": p.qty,
                    "fees_paid": p.fees_paid,
                    "realized_pnl": p.realized_pnl,
                    "meta": p.meta,
                }
                for p in positions
            ]
        )

    @staticmethod
    def trades_to_pl(trades: Iterable[Trade]) -> pl.DataFrame:
        """Convert trades to Polars DataFrame.

        Args:
            trades (Iterable[Trade]): Trades to convert.

        Returns:
            pl.DataFrame: DataFrame with trade data. Empty if no trades.

        Example:
            ```python
            # Convert all trades
            trades_df = Portfolio.trades_to_pl(all_trades)

            # Analyze trades
            print(trades_df.select(["pair", "side", "price", "qty"]))

            # Filter by type
            entry_trades = trades_df.filter(
                pl.col("meta").struct.field("type") == "entry"
            )

            # Calculate total volume
            total_volume = trades_df.select(
                (pl.col("price") * pl.col("qty")).sum()
            ).item()
            ```
        """
        if not trades:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "id": t.id,
                    "position_id": t.position_id,
                    "pair": t.pair,
                    "side": t.side,
                    "ts": t.ts,
                    "price": t.price,
                    "qty": t.qty,
                    "fee": t.fee,
                    "meta": t.meta,
                }
                for t in trades
            ]
        )
