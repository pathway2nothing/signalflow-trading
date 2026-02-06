"""Unlimited balance runner - no balance constraints for maximum speed."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar
import uuid

import polars as pl
from loguru import logger
from tqdm import tqdm

from signalflow.core.enums import SfComponentType
from signalflow.core.decorators import sf_component
from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.containers.order import Order
from signalflow.strategy.component.base import EntryRule
from signalflow.strategy.runner.base import StrategyRunner
from signalflow.strategy.runner.parallel.results import UnlimitedResults
from signalflow.strategy.broker.unlimited_broker import UnlimitedBacktestBroker
from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
from signalflow.data.strategy_store.memory import InMemoryStrategyStore


@dataclass
@sf_component(name="unlimited_balance_runner", override=True)
class UnlimitedBalanceRunner(StrategyRunner):
    """Simplified backtest runner with unlimited balance.

    Key characteristics:
    - No balance constraints (always enough funds)
    - Fixed position size for all trades
    - Simple TP/SL exit logic (no custom exit_rules)
    - Maximum speed through simplified logic

    Use cases:
    - Quick signal validation
    - A/B testing of entry rules
    - Hit rate analysis without capital constraints

    Attributes:
        strategy_id: Identifier for this strategy
        broker: Created internally (UnlimitedBacktestBroker)
        entry_rules: Entry rule instances
        position_size: Fixed position size in quote currency
        take_profit_pct: Take profit percentage (e.g., 0.02 = 2%)
        stop_loss_pct: Stop loss percentage (e.g., 0.01 = 1%)
        max_bars_in_trade: Maximum bars to hold position (None = no limit)
        pair_col: Column name for pair identifier
        ts_col: Column name for timestamp
        price_col: Column name for price
        data_key: Key in RawData for OHLCV data
        fee_rate: Trading fee rate
        show_progress: Show progress bar
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_RUNNER

    strategy_id: str = "unlimited_backtest"
    broker: Any = None
    entry_rules: list[EntryRule] = field(default_factory=list)
    exit_rules: list = field(default_factory=list)  # Ignored - uses TP/SL
    metrics: list = field(default_factory=list)

    position_size: float = 1.0
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    max_bars_in_trade: int | None = None

    pair_col: str = "pair"
    ts_col: str = "timestamp"
    price_col: str = "close"
    data_key: str = "spot"
    fee_rate: float = 0.001
    show_progress: bool = True

    _trades: list[Trade] = field(default_factory=list, init=False)

    def run(
        self, raw_data: RawData, signals: Signals, state: StrategyState | None = None
    ) -> UnlimitedResults:
        """Run unlimited balance backtest.

        Args:
            raw_data: Historical OHLCV data
            signals: Pre-computed signals
            state: Ignored - creates fresh state

        Returns:
            UnlimitedResults with trade statistics
        """
        df = raw_data.get(self.data_key)
        signals_df = signals.value if signals else pl.DataFrame()

        if df.height == 0:
            logger.warning("No data to backtest")
            return self._empty_result()

        if signals_df.height == 0:
            logger.warning("No signals to process")
            return self._empty_result()

        logger.info(
            f"Starting unlimited backtest: {signals_df.height} signals, "
            f"TP={self.take_profit_pct:.1%}, SL={self.stop_loss_pct:.1%}"
        )

        # Create state and broker
        state = StrategyState(strategy_id=self.strategy_id)
        broker = UnlimitedBacktestBroker(
            executor=VirtualSpotExecutor(fee_rate=self.fee_rate),
            store=InMemoryStrategyStore(),
        )

        # Get timestamps
        timestamps = df.select(self.ts_col).unique().sort(self.ts_col).get_column(self.ts_col).to_list()

        self._trades = []
        position_entry_bars: dict[str, int] = {}  # position_id -> entry bar index

        iterator = enumerate(timestamps)
        if self.show_progress:
            iterator = tqdm(list(iterator), desc="Processing bars")

        for bar_idx, ts in iterator:
            state.touch(ts)
            state.reset_tick_cache()

            # Get bar data
            bar_data = df.filter(pl.col(self.ts_col) == ts)
            prices = self._build_prices(bar_data)

            # Mark positions
            broker.mark_positions(state, prices, ts)

            # Check TP/SL exits
            exit_orders = self._check_tp_sl_exits(state, prices, bar_idx, position_entry_bars)
            if exit_orders:
                exit_fills = broker.submit_orders(exit_orders, prices, ts)
                exit_trades = broker.process_fills(exit_fills, exit_orders, state)
                self._trades.extend(exit_trades)
                # Remove closed positions from tracking
                for trade in exit_trades:
                    if trade.position_id in position_entry_bars:
                        del position_entry_bars[trade.position_id]

            # Get bar signals
            bar_signals_df = signals_df.filter(pl.col(self.ts_col) == ts)
            bar_signals = Signals(bar_signals_df)

            # Check entries
            entry_orders = []
            for entry_rule in self.entry_rules:
                orders = entry_rule.check_entries(bar_signals, prices, state)
                entry_orders.extend(orders)

            if entry_orders:
                entry_fills = broker.submit_orders(entry_orders, prices, ts)
                entry_trades = broker.process_fills(entry_fills, entry_orders, state)
                self._trades.extend(entry_trades)
                # Track entry bar for each position
                for trade in entry_trades:
                    if trade.meta.get("type") == "entry":
                        position_entry_bars[trade.position_id] = bar_idx

        # Build results
        return self._build_results(signals_df.height)

    def _build_prices(self, bar_data: pl.DataFrame) -> dict[str, float]:
        """Build pair -> price mapping."""
        prices = {}
        for row in bar_data.iter_rows(named=True):
            pair = row.get(self.pair_col)
            price = row.get(self.price_col)
            if pair and price is not None:
                prices[pair] = float(price)
        return prices

    def _check_tp_sl_exits(
        self,
        state: StrategyState,
        prices: dict[str, float],
        current_bar: int,
        position_entry_bars: dict[str, int],
    ) -> list[Order]:
        """Check TP/SL conditions for open positions."""
        orders = []

        for position in state.portfolio.open_positions():
            price = prices.get(position.pair)
            if price is None:
                continue

            should_exit = False
            exit_reason = ""

            # Calculate return
            if position.position_type.value == "long":
                pct_return = (price - position.entry_price) / position.entry_price
                if pct_return >= self.take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"
                elif pct_return <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:  # short
                pct_return = (position.entry_price - price) / position.entry_price
                if pct_return >= self.take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"
                elif pct_return <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"

            # Check max bars in trade
            if self.max_bars_in_trade is not None:
                entry_bar = position_entry_bars.get(position.id, 0)
                if current_bar - entry_bar >= self.max_bars_in_trade:
                    should_exit = True
                    exit_reason = "max_bars"

            if should_exit:
                # Create exit order
                side = "SELL" if position.position_type.value == "long" else "BUY"
                order = Order(
                    id=str(uuid.uuid4()),
                    pair=position.pair,
                    side=side,
                    qty=position.qty,
                    position_id=position.id,
                    meta={"reason": exit_reason},
                )
                orders.append(order)

        return orders

    def _build_results(self, total_signals: int) -> UnlimitedResults:
        """Build results from trades."""
        if not self._trades:
            return UnlimitedResults(
                trades_df=pl.DataFrame(),
                total_signals=total_signals,
                executed_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                hit_rate=0.0,
            )

        # Separate entry and exit trades
        entry_trades = [t for t in self._trades if t.meta.get("type") == "entry"]
        exit_trades = [t for t in self._trades if t.meta.get("type") == "exit"]

        # Build trades DataFrame
        trade_rows = []
        exit_by_position = {t.position_id: t for t in exit_trades}

        for entry in entry_trades:
            exit_trade = exit_by_position.get(entry.position_id)
            if exit_trade:
                entry_price = entry.price
                exit_price = exit_trade.price

                # Determine side from entry trade
                is_long = entry.side == "BUY"

                if is_long:
                    pnl = (exit_price - entry_price) * entry.qty
                    return_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) * entry.qty
                    return_pct = (entry_price - exit_price) / entry_price

                # Subtract fees
                total_fees = entry.fee + exit_trade.fee
                pnl -= total_fees

                trade_rows.append({
                    "pair": entry.pair,
                    "side": "LONG" if is_long else "SHORT",
                    "entry_time": entry.ts,
                    "exit_time": exit_trade.ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": entry.qty,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "exit_reason": exit_trade.meta.get("reason", "unknown"),
                    "fees": total_fees,
                })

        trades_df = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame()

        # Calculate statistics
        executed_trades = len(trade_rows)
        if executed_trades > 0:
            win_trades = len([r for r in trade_rows if r["pnl"] > 0])
            win_rate = win_trades / executed_trades
            avg_return = sum(r["return_pct"] for r in trade_rows) / executed_trades
            hit_rate = len([r for r in trade_rows if r["exit_reason"] == "take_profit"]) / executed_trades
        else:
            win_rate = 0.0
            avg_return = 0.0
            hit_rate = 0.0

        logger.info(
            f"Unlimited backtest complete: {executed_trades} trades, "
            f"win_rate={win_rate:.1%}, avg_return={avg_return:.2%}"
        )

        return UnlimitedResults(
            trades_df=trades_df,
            total_signals=total_signals,
            executed_trades=executed_trades,
            win_rate=win_rate,
            avg_return=avg_return,
            hit_rate=hit_rate,
        )

    def _empty_result(self) -> UnlimitedResults:
        """Return empty result."""
        return UnlimitedResults(
            trades_df=pl.DataFrame(),
            total_signals=0,
            executed_trades=0,
            win_rate=0.0,
            avg_return=0.0,
            hit_rate=0.0,
        )

    @property
    def trades(self) -> list[Trade]:
        """Get all trades from the backtest."""
        return self._trades

    @property
    def trades_df(self) -> pl.DataFrame:
        """Get trades as a DataFrame."""
        from signalflow.core.containers.portfolio import Portfolio

        return Portfolio.trades_to_pl(self._trades)

    def get_results(self) -> dict[str, Any]:
        """Get backtest results summary."""
        return {
            "total_trades": len(self._trades),
            "trades_df": self.trades_df,
        }
