from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar
from signalflow.core import SfComponentType, StrategyState, Signals, sf_component
from signalflow.strategy.component.base import EntryRule, ExitRule, StrategyMetric
from signalflow.strategy.runner.base import StrategyRunner
from datetime import datetime
import polars as pl

@dataclass
@sf_component(name='optimized_backtest_runner', override=True)
class OptimizedBacktestRunner(StrategyRunner):
    """
    Оптимізований runner з:
    - Vectorized price lookup
    - Batch processing
    - Pre-filtered signals
    """
    component_type = SfComponentType.STRATEGY_RUNNER
    strategy_id: str = 'backtest'
    broker: Any = None
    entry_rules: list[EntryRule] = field(default_factory=list)
    exit_rules: list = field(default_factory=list)
    metrics: list[StrategyMetric] = field(default_factory=list)
    initial_capital: float = 10000.0
    pair_col: str = 'pair'
    ts_col: str = 'timestamp'
    price_col: str = 'close'
    data_key: str = 'spot'
    show_progress: bool = True
    _trades: list = field(default_factory=list, init=False)
    _metrics_history: list[dict] = field(default_factory=list, init=False)

    def run(self, raw_data, signals: Signals, state: StrategyState | None = None):
        from signalflow.core.containers.strategy_state import StrategyState
        from tqdm import tqdm
        
        if state is None:
            state = StrategyState(strategy_id=self.strategy_id)
            state.portfolio.cash = self.initial_capital
        
        self._trades = []
        self._metrics_history = []
        
        df = raw_data.get(self.data_key)
        if df.height == 0:
            return state
        
        timestamps = df.select(self.ts_col).unique().sort(self.ts_col).get_column(self.ts_col)
        signals_df = signals.value if signals else pl.DataFrame()
        
        price_lookup = self._build_price_lookup(df)
        
        signal_lookup = self._build_signal_lookup(signals_df) if signals_df.height > 0 else {}
        
        iterator = tqdm(timestamps, desc='Backtesting') if self.show_progress else timestamps
        
        for ts in iterator:
            state = self._process_bar_optimized(
                ts=ts,
                price_lookup=price_lookup,
                signal_lookup=signal_lookup,
                state=state
            )
        
        return state

    def _build_price_lookup(self, df: pl.DataFrame) -> dict[datetime, dict[str, float]]:
        """Створює lookup таблицю: timestamp -> {pair: price}"""
        lookup = {}
        for row in df.select([self.ts_col, self.pair_col, self.price_col]).iter_rows():
            ts, pair, price = row
            if ts not in lookup:
                lookup[ts] = {}
            lookup[ts][pair] = float(price)
        return lookup

    def _build_signal_lookup(self, signals_df: pl.DataFrame) -> dict[datetime, pl.DataFrame]:
        """Створює lookup: timestamp -> signals DataFrame"""
        lookup = {}
        if signals_df.height == 0:
            return lookup
        
        for ts in signals_df.select(self.ts_col).unique().get_column(self.ts_col):
            lookup[ts] = signals_df.filter(pl.col(self.ts_col) == ts)
        
        return lookup

    def _process_bar_optimized(
        self,
        ts: datetime,
        price_lookup: dict[datetime, dict[str, float]],
        signal_lookup: dict[datetime, pl.DataFrame],
        state: StrategyState
    ) -> StrategyState:
        state.touch(ts)
        state.reset_tick_cache()
        
        prices = price_lookup.get(ts, {})
        
        self.broker.mark_positions(state, prices, ts)
        
        all_metrics: dict[str, float] = {'timestamp': ts.timestamp()}
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
        
        bar_signals_df = signal_lookup.get(ts, pl.DataFrame())
        bar_signals = Signals(bar_signals_df)
        
        entry_orders = []
        for entry_rule in self.entry_rules:
            orders = entry_rule.check_entries(bar_signals, prices, state)
            entry_orders.extend(orders)
        
        if entry_orders:
            entry_fills = self.broker.submit_orders(entry_orders, prices, ts)
            entry_trades = self.broker.process_fills(entry_fills, entry_orders, state)
            self._trades.extend(entry_trades)
        
        return state

    @property
    def trades(self):
        return self._trades

    @property
    def trades_df(self) -> pl.DataFrame:
        from signalflow.core.containers.portfolio import Portfolio
        return Portfolio.trades_to_pl(self._trades)

    @property
    def metrics_df(self) -> pl.DataFrame:
        if not self._metrics_history:
            return pl.DataFrame()
        return pl.DataFrame(self._metrics_history)

    def get_results(self) -> dict[str, Any]:
        trades_df = self.trades_df
        metrics_df = self.metrics_df
        
        results = {
            'total_trades': len(self._trades),
            'metrics_df': metrics_df,
            'trades_df': trades_df,
        }
        
        if metrics_df.height > 0:
            last_row = metrics_df.tail(1)
            
            if 'total_return' in metrics_df.columns:
                results['final_return'] = last_row.select('total_return').item()
            if 'equity' in metrics_df.columns:
                results['final_equity'] = last_row.select('equity').item()
            if 'sharpe_ratio' in metrics_df.columns:
                results['sharpe_ratio'] = last_row.select('sharpe_ratio').item()
            if 'max_drawdown' in metrics_df.columns:
                results['max_drawdown'] = last_row.select('max_drawdown').item()
            if 'win_rate' in metrics_df.columns:
                results['win_rate'] = last_row.select('win_rate').item()
        
        if trades_df.height > 0:
            entry_trades = trades_df.filter(
                pl.col('meta').struct.field('type') == 'entry'
            )
            exit_trades = trades_df.filter(
                pl.col('meta').struct.field('type') == 'exit'
            )
            results['entry_count'] = entry_trades.height
            results['exit_count'] = exit_trades.height
        
        return results