"""Backtest runner for strategy execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from loguru import logger

import polars as pl

from signalflow.core.containers import Position, Trade, Portfolio
from signalflow.strategy.state import StrategyState
from signalflow.strategy.types import StrategyContext, NewPositionOrder, ClosePositionOrder
from signalflow.strategy.component.base import StrategyMetric, StrategyEntryRule, StrategyExitRule
from signalflow.strategy.executor.backtest import BacktestExecutor
from signalflow.strategy.signal_source import PrecomputedSignalSource, OhlcvPriceSource


@dataclass
class BacktestResult:
    """Results from backtest run.
    
    Attributes:
        positions: All positions (open and closed)
        trades: All executed trades
        metrics_history: Time series of metrics
        final_state: Final strategy state
    """
    positions: list[Position]
    trades: list[Trade]
    metrics_history: list[dict[str, Any]]
    final_state: StrategyState
    
    def positions_df(self) -> pl.DataFrame:
        """Convert positions to DataFrame."""
        return Portfolio.positions_to_pl(self.positions)
    
    def trades_df(self) -> pl.DataFrame:
        """Convert trades to DataFrame."""
        return Portfolio.trades_to_pl(self.trades)
    
    def metrics_df(self) -> pl.DataFrame:
        """Convert metrics history to DataFrame."""
        if not self.metrics_history:
            return pl.DataFrame()
        return pl.DataFrame(self.metrics_history)
    
    @property
    def total_pnl(self) -> float:
        """Total realized PnL across all closed positions."""
        return sum(p.realized_pnl for p in self.positions if p.is_closed)
    
    @property
    def total_fees(self) -> float:
        """Total fees paid."""
        return sum(p.fees_paid for p in self.positions)
    
    @property
    def net_pnl(self) -> float:
        """Net PnL after fees."""
        return self.total_pnl - self.total_fees
    
    @property
    def num_trades(self) -> int:
        """Number of completed trades (closed positions)."""
        return sum(1 for p in self.positions if p.is_closed)
    
    @property
    def win_rate(self) -> float:
        """Win rate (profitable trades / total trades)."""
        closed = [p for p in self.positions if p.is_closed]
        if not closed:
            return 0.0
        wins = sum(1 for p in closed if p.realized_pnl > 0)
        return wins / len(closed)


@dataclass
class BacktestRunner:
    """Backtest runner for strategy execution.
    
    Processes signals chronologically, executing entries and exits
    according to configured rules.
    
    Step order:
    1. Mark prices (update last_price for all open positions)
    2. Compute metrics (available to entry/exit rules)
    3. Check & execute exits
    4. Check & execute entries
    
    Attributes:
        strategy_id: Unique identifier for this strategy run
        signal_source: Source of trading signals
        price_source: Source of prices
        entry_rule: Rule for generating entry orders
        exit_rules: List of exit rules (checked in order)
        metrics: List of metrics to compute each step
        executor: Order executor (default: BacktestExecutor)
    """
    strategy_id: str
    signal_source: PrecomputedSignalSource
    price_source: OhlcvPriceSource
    entry_rule: StrategyEntryRule
    exit_rules: list[StrategyExitRule] = field(default_factory=list)
    metrics: list[StrategyMetric] = field(default_factory=list)
    executor: BacktestExecutor = field(default_factory=BacktestExecutor)
    
    # Callbacks for monitoring
    on_step: Callable[[StrategyState, StrategyContext], None] | None = None
    on_entry: Callable[[Position, Trade], None] | None = None
    on_exit: Callable[[Position, Trade], None] | None = None
    
    def run(
        self,
        initial_state: StrategyState | None = None,
        progress: bool = True,
    ) -> BacktestResult:
        """Run backtest over all signals.
        
        Args:
            initial_state: Starting state (default: empty portfolio)
            progress: Whether to log progress
            
        Returns:
            BacktestResult with all positions, trades, and metrics
        """
        state = initial_state or StrategyState(
            strategy_id=self.strategy_id,
            portfolio=Portfolio(cash=0.0),
        )
        
        all_trades: list[Trade] = []
        metrics_history: list[dict[str, Any]] = []
        
        timestamps = list(self.signal_source.timestamps())
        total = len(timestamps)
        
        if progress:
            logger.info(f"Starting backtest with {total} signal timestamps")
        
        for i, ts in enumerate(timestamps):
            prices = self.price_source.get_prices_at(ts)
            if not prices:
                continue
            
            signals_df = self.signal_source.get_signals_at(ts)
            
            step_trades, step_metrics = self._step(
                state=state,
                ts=ts,
                prices=prices,
                signals_df=signals_df,
            )
            
            all_trades.extend(step_trades)
            if step_metrics:
                metrics_history.append({'timestamp': ts, **step_metrics})
            
            if progress and (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{total} timestamps")
        
        if progress:
            logger.info(f"Backtest complete. {len(all_trades)} trades executed.")
        
        all_positions = list(state.portfolio.positions.values())
        
        return BacktestResult(
            positions=all_positions,
            trades=all_trades,
            metrics_history=metrics_history,
            final_state=state,
        )
    
    def _step(
        self,
        state: StrategyState,
        ts: datetime,
        prices: dict[str, float],
        signals_df: pl.DataFrame,
    ) -> tuple[list[Trade], dict[str, float]]:
        """Execute single step.
        
        Returns:
            Tuple of (trades executed, metrics computed)
        """
        trades: list[Trade] = []
        
        for position in state.portfolio.open_positions():
            price = prices.get(position.pair)
            if price is not None:
                position.mark(ts=ts, price=float(price))
        
        computed_metrics: dict[str, float] = {}
        for metric in self.metrics:
            ctx = StrategyContext(
                strategy_id=self.strategy_id,
                ts=ts,
                prices=prices,
                metrics={},  
                runtime=state.runtime,
            )
            computed_metrics.update(metric.compute(state=state, context=ctx))
        
        context = StrategyContext(
            strategy_id=self.strategy_id,
            ts=ts,
            prices=prices,
            metrics=computed_metrics,
            runtime=state.runtime,
        )
        
        exit_trades = self._process_exits(state, context)
        trades.extend(exit_trades)
        
        entry_trades = self._process_entries(state, context, signals_df)
        trades.extend(entry_trades)
        
        state.last_ts = ts
        
        if self.on_step:
            self.on_step(state, context)
        
        return trades, computed_metrics
    
    def _process_exits(
        self,
        state: StrategyState,
        context: StrategyContext,
    ) -> list[Trade]:
        """Check and execute exits for open positions."""
        trades: list[Trade] = []
        
        for position in list(state.portfolio.open_positions()):
            for exit_rule in self.exit_rules:
                should_exit, reason = exit_rule.should_exit(
                    position=position,
                    state=state,
                    context=context,
                )
                
                if should_exit:
                    price = context.prices.get(position.pair)
                    if price is None:
                        logger.warning(
                            f"No price for {position.pair} at {context.ts}, "
                            f"skipping exit"
                        )
                        continue
                    
                    order = ClosePositionOrder(
                        position_id=position.id,
                        ts=context.ts,
                        price=float(price),
                        reason=reason,
                    )
                    
                    trade = self.executor.execute_exit(position, order)
                    trades.append(trade)
                    
                    if self.on_exit:
                        self.on_exit(position, trade)
                    
                    break
        
        return trades
    
    def _process_entries(
        self,
        state: StrategyState,
        context: StrategyContext,
        signals_df: pl.DataFrame,
    ) -> list[Trade]:
        """Check and execute entries based on signals."""
        trades: list[Trade] = []
        
        if signals_df.is_empty():
            return trades
        
        # Get entry orders from rule
        orders = self.entry_rule.build_orders(
            signals=signals_df,
            state=state,
            context=context,
        )
        
        for order in orders:
            # Execute entry
            position, trade = self.executor.execute_entry(order)
            
            # Add position to portfolio
            state.portfolio.positions[position.id] = position
            trades.append(trade)
            
            # Callback
            if self.on_entry:
                self.on_entry(position, trade)
        
        return trades