"""Isolated balance runner - parallel processing with isolated balance per pair."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl
from loguru import logger
from tqdm import tqdm

from signalflow.core.enums import SfComponentType
from signalflow.core.decorators import sf_component
from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.strategy.component.base import EntryRule, ExitRule
from signalflow.strategy.runner.base import StrategyRunner
from signalflow.strategy.runner.parallel.results import PairResult, IsolatedResults
from signalflow.analytic import StrategyMetric


def _run_pair_backtest(
    pair: str,
    pair_df: pl.DataFrame,
    pair_signals: pl.DataFrame,
    pair_capital: float,
    config: dict,
    entry_rules: list,
    exit_rules: list,
    metrics: list,
) -> PairResult:
    """Run backtest for single pair."""
    from signalflow.strategy.broker.isolated_broker import IsolatedBacktestBroker
    from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
    from signalflow.data.strategy_store.memory import InMemoryStrategyStore

    # Create isolated state
    state = StrategyState(strategy_id=f"{config['strategy_id']}_{pair}")
    state.portfolio.cash = pair_capital

    # Create broker for this pair
    broker = IsolatedBacktestBroker(
        pair=pair,
        executor=VirtualSpotExecutor(fee_rate=config.get("fee_rate", 0.001)),
        store=InMemoryStrategyStore(),
    )

    # Get column names from config
    ts_col = config.get("ts_col", "timestamp")
    price_col = config.get("price_col", "close")
    pair_col = config.get("pair_col", "pair")

    # Get timestamps
    timestamps = pair_df.select(ts_col).unique().sort(ts_col).get_column(ts_col).to_list()

    trades: list[Trade] = []
    metrics_history: list[dict] = []

    for ts in timestamps:
        state.touch(ts)
        state.reset_tick_cache()

        # Get bar data
        bar_data = pair_df.filter(pl.col(ts_col) == ts)
        prices = {}
        for row in bar_data.iter_rows(named=True):
            p = row.get(pair_col)
            price = row.get(price_col)
            if p and price is not None:
                prices[p] = float(price)

        # Mark positions
        broker.mark_positions(state, prices, ts)

        # Compute metrics
        all_metrics: dict[str, float] = {"timestamp": ts.timestamp() if hasattr(ts, "timestamp") else float(ts)}
        for metric in metrics:
            metric_values = metric.compute(state, prices)
            all_metrics.update(metric_values)
        state.metrics = all_metrics
        metrics_history.append(all_metrics.copy())

        # Check exits
        exit_orders = []
        open_positions = state.portfolio.open_positions()
        for exit_rule in exit_rules:
            orders = exit_rule.check_exits(open_positions, prices, state)
            exit_orders.extend(orders)

        if exit_orders:
            exit_fills = broker.submit_orders(exit_orders, prices, ts)
            exit_trades = broker.process_fills(exit_fills, exit_orders, state)
            trades.extend(exit_trades)

        # Get bar signals
        if pair_signals.height > 0:
            bar_signals_df = pair_signals.filter(pl.col(ts_col) == ts)
            bar_signals = Signals(bar_signals_df)
        else:
            bar_signals = Signals(pl.DataFrame())

        # Check entries
        entry_orders = []
        for entry_rule in entry_rules:
            orders = entry_rule.check_entries(bar_signals, prices, state)
            entry_orders.extend(orders)

        if entry_orders:
            entry_fills = broker.submit_orders(entry_orders, prices, ts)
            entry_trades = broker.process_fills(entry_fills, entry_orders, state)
            trades.extend(entry_trades)

    # Calculate final equity
    final_prices = {}
    if timestamps:
        last_bar = pair_df.filter(pl.col(ts_col) == timestamps[-1])
        for row in last_bar.iter_rows(named=True):
            p = row.get(pair_col)
            price = row.get(price_col)
            if p and price is not None:
                final_prices[p] = float(price)

    final_equity = state.portfolio.equity(prices=final_prices)

    return PairResult(
        pair=pair,
        trades=trades,
        final_equity=final_equity,
        final_cash=state.portfolio.cash,
        positions=list(state.portfolio.positions.values()),
        metrics_history=metrics_history,
        initial_capital=pair_capital,
    )


@dataclass
@sf_component(name="isolated_balance_runner", override=True)
class IsolatedBalanceRunner(StrategyRunner):
    """Parallel backtest runner with isolated balance per pair.

    Splits initial capital equally among pairs and runs each
    pair independently using ThreadPoolExecutor.

    Attributes:
        strategy_id: Identifier for this strategy
        broker: Not used directly - creates IsolatedBacktestBroker per pair
        entry_rules: Entry rule instances
        exit_rules: Exit rule instances
        metrics: Metric instances
        initial_capital: Total starting capital (split among pairs)
        max_workers: Number of parallel workers (None = cpu_count)
        pair_col: Column name for pair identifier
        ts_col: Column name for timestamp
        price_col: Column name for price
        data_key: Key in RawData for OHLCV data
        fee_rate: Trading fee rate
        show_progress: Show progress bar
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_RUNNER

    strategy_id: str = "isolated_backtest"
    broker: Any = None
    entry_rules: list[EntryRule] = field(default_factory=list)
    exit_rules: list[ExitRule] = field(default_factory=list)
    metrics: list[StrategyMetric] = field(default_factory=list)

    initial_capital: float = 10000.0
    max_workers: int | None = None
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    price_col: str = "close"
    data_key: str = "spot"
    fee_rate: float = 0.001
    show_progress: bool = True

    _trades: list[Trade] = field(default_factory=list, init=False)
    _metrics_history: list[dict] = field(default_factory=list, init=False)

    def run(
        self, raw_data: RawData, signals: Signals, state: StrategyState | None = None
    ) -> IsolatedResults:
        """Run parallel backtest with isolated balance per pair.

        Args:
            raw_data: Historical OHLCV data
            signals: Pre-computed signals
            state: Ignored - creates fresh state per pair

        Returns:
            IsolatedResults with aggregated and per-pair results
        """
        _ = state  # Unused - each pair gets fresh state

        df = raw_data.get(self.data_key)
        if df.height == 0:
            logger.warning("No data to backtest")
            return IsolatedResults(
                total_equity=self.initial_capital,
                total_return=0.0,
                initial_capital=self.initial_capital,
                pair_results={},
            )

        # Get unique pairs
        pairs = df.select(self.pair_col).unique().get_column(self.pair_col).to_list()
        pair_capital = self.initial_capital / len(pairs)

        signals_df = signals.value if signals else pl.DataFrame()

        logger.info(
            f"Starting isolated backtest: {len(pairs)} pairs, "
            f"{pair_capital:.2f} capital per pair, {self.max_workers or 'auto'} workers"
        )

        # Prepare config
        config = {
            "strategy_id": self.strategy_id,
            "pair_col": self.pair_col,
            "ts_col": self.ts_col,
            "price_col": self.price_col,
            "fee_rate": self.fee_rate,
        }

        # Prepare tasks
        tasks = []
        for pair in pairs:
            pair_df = df.filter(pl.col(self.pair_col) == pair)
            pair_signals = (
                signals_df.filter(pl.col(self.pair_col) == pair)
                if signals_df.height > 0 and self.pair_col in signals_df.columns
                else pl.DataFrame()
            )
            tasks.append((pair, pair_df, pair_signals, pair_capital))

        # Run in parallel using ThreadPoolExecutor
        pair_results: list[PairResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _run_pair_backtest,
                    pair,
                    pair_df,
                    pair_signals,
                    pair_capital,
                    config,
                    self.entry_rules,
                    self.exit_rules,
                    self.metrics,
                ): pair
                for pair, pair_df, pair_signals, pair_capital in tasks
            }

            iterator = as_completed(futures)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Processing pairs")

            for future in iterator:
                pair = futures[future]
                try:
                    result = future.result()
                    pair_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing pair {pair}: {e}")
                    raise

        # Aggregate results
        return self._aggregate_results(pair_results)

    def _aggregate_results(self, pair_results: list[PairResult]) -> IsolatedResults:
        """Aggregate results from all pairs."""
        if not pair_results:
            return IsolatedResults(
                total_equity=self.initial_capital,
                total_return=0.0,
                initial_capital=self.initial_capital,
                pair_results={},
            )

        total_equity = sum(r.final_equity for r in pair_results)
        total_return = (total_equity - self.initial_capital) / self.initial_capital

        # Collect all trades and metrics
        self._trades = []
        self._metrics_history = []
        for r in pair_results:
            self._trades.extend(r.trades)
            self._metrics_history.extend(r.metrics_history)

        logger.info(
            f"Isolated backtest complete: {len(pair_results)} pairs, "
            f"{len(self._trades)} trades, return={total_return:.2%}"
        )

        return IsolatedResults(
            total_equity=total_equity,
            total_return=total_return,
            initial_capital=self.initial_capital,
            pair_results={r.pair: r for r in pair_results},
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

        return {
            "total_trades": len(self._trades),
            "metrics_df": metrics_df,
            "trades_df": trades_df,
        }
