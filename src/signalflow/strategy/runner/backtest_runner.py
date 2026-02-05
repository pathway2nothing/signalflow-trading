"""Backtest runner - orchestrates the backtesting loop."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import polars as pl
from loguru import logger

from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.decorators import sf_component
from signalflow.strategy.component.base import EntryRule, ExitRule
from tqdm import tqdm
from signalflow.strategy.runner.base import StrategyRunner
from signalflow.analytic import StrategyMetric


@dataclass
@sf_component(name="backtest_runner")
class BacktestRunner(StrategyRunner):
    """
    Runs backtests over historical data.

    Execution flow per bar:
        1. Mark prices on all positions
        2. Compute metrics
        3. Check and execute exits
        4. Check and execute entries

    This order ensures:
        - Metrics reflect current market state
        - Exits are processed before entries (can close and re-enter same bar)
        - No look-ahead bias
    """

    strategy_id: str = "backtest"
    broker: Any = None
    entry_rules: list[EntryRule] = field(default_factory=list)
    exit_rules: list[ExitRule] = field(default_factory=list)
    metrics: list[StrategyMetric] = field(default_factory=list)

    initial_capital: float = 10000.0
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    price_col: str = "close"
    data_key: str = "spot"

    _trades: list[Trade] = field(default_factory=list, init=False)
    _metrics_history: list[dict] = field(default_factory=list, init=False)

    def run(self, raw_data: RawData, signals: Signals, state: StrategyState | None = None) -> StrategyState:
        """
        Run backtest over the entire dataset.

        Args:
            raw_data: Historical OHLCV data
            signals: Pre-computed signals for the period
            state: Optional initial state (for continuing backtests)

        Returns:
            Final strategy state
        """
        if state is None:
            state = StrategyState(
                strategy_id=self.strategy_id,
            )
            state.portfolio.cash = self.initial_capital

        self._trades = []
        self._metrics_history = []

        # Get data
        df = raw_data.get(self.data_key)
        if df.height == 0:
            logger.warning("No data to backtest")
            return state

        timestamps = df.select(self.ts_col).unique().sort(self.ts_col).get_column(self.ts_col)

        signals_df = signals.value if signals else pl.DataFrame()

        logger.info(f"Starting backtest: {len(timestamps)} bars, {signals_df.height} signals")

        for ts in tqdm(timestamps, desc="Processing bars"):
            state = self._process_bar(ts=ts, raw_df=df, signals_df=signals_df, state=state)

        logger.info(
            f"Backtest complete: {len(self._trades)} trades, {len(state.portfolio.open_positions())} open positions"
        )

        return state

    def _process_bar(
        self, ts: datetime, raw_df: pl.DataFrame, signals_df: pl.DataFrame, state: StrategyState
    ) -> StrategyState:
        """Process a single bar."""
        state.touch(ts)
        state.reset_tick_cache()

        bar_data = raw_df.filter(pl.col(self.ts_col) == ts)
        prices = self._build_prices(bar_data)
        self.broker.mark_positions(state, prices, ts)

        all_metrics: dict[str, float] = {"timestamp": ts.timestamp()}
        for metric in self.metrics:
            metric_values = metric.compute(state, prices)
            all_metrics.update(metric_values)
        state.metrics = all_metrics
        self._metrics_history.append(all_metrics.copy())

        exit_orders = []
        open_positions = state.portfolio.open_positions()
        for exit_rule in self.exit_rules:
            orders = exit_rule.check_exits(open_positions, prices, state)
            exit_orders.extend(orders)

        if exit_orders:
            exit_fills = self.broker.submit_orders(exit_orders, prices, ts)
            exit_trades = self.broker.process_fills(exit_fills, exit_orders, state)
            self._trades.extend(exit_trades)

        bar_signals = self._get_bar_signals(signals_df, ts)

        entry_orders = []
        for entry_rule in self.entry_rules:
            orders = entry_rule.check_entries(bar_signals, prices, state)
            entry_orders.extend(orders)

        if entry_orders:
            entry_fills = self.broker.submit_orders(entry_orders, prices, ts)
            entry_trades = self.broker.process_fills(entry_fills, entry_orders, state)
            self._trades.extend(entry_trades)

        return state

    def _build_prices(self, bar_data: pl.DataFrame) -> dict[str, float]:
        """Build pair -> price mapping from bar data."""
        prices = {}
        for row in bar_data.iter_rows(named=True):
            pair = row.get(self.pair_col)
            price = row.get(self.price_col)
            if pair and price is not None:
                prices[pair] = float(price)
        return prices

    def _get_bar_signals(self, signals_df: pl.DataFrame, ts: datetime) -> Signals:
        """Get signals for the current bar."""
        if signals_df.height == 0:
            return Signals(pl.DataFrame())

        bar_signals = signals_df.filter(pl.col(self.ts_col) == ts)
        return Signals(bar_signals)

    @property
    def trades(self) -> list[Trade]:
        """Get all trades from the backtest."""
        return self._trades

    @property
    def trades_df(self) -> pl.DataFrame:
        """Get trades as a DataFrame."""
        from signalflow.core.containers.portfolio import Portfolio

        return Portfolio.trades_to_pl(self._trades)

    @property
    def metrics_df(self) -> pl.DataFrame:
        """Get metrics history as a DataFrame."""
        if not self._metrics_history:
            return pl.DataFrame()
        return pl.DataFrame(self._metrics_history)

    def get_results(self) -> dict[str, Any]:
        """Get backtest results summary."""
        trades_df = self.trades_df
        metrics_df = self.metrics_df

        results = {
            "total_trades": len(self._trades),
            "metrics_df": metrics_df,
            "trades_df": trades_df,
        }

        if metrics_df.height > 0 and "total_return" in metrics_df.columns:
            results["final_return"] = metrics_df.select("total_return").tail(1).item()
            results["final_equity"] = metrics_df.select("equity").tail(1).item()

        if trades_df.height > 0:
            entry_trades = trades_df.filter(pl.col("meta").struct.field("type") == "entry")
            exit_trades = trades_df.filter(pl.col("meta").struct.field("type") == "exit")
            results["entry_count"] = entry_trades.height
            results["exit_count"] = exit_trades.height

        return results
