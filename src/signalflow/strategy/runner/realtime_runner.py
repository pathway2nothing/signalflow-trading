"""Realtime runner - async paper/live trading loop."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import polars as pl
from loguru import logger

from signalflow.analytic import StrategyMetric
from signalflow.core.containers.raw_data import RawData
from signalflow.core.containers.raw_data_view import RawDataView
from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.containers.trade import Trade
from signalflow.core.decorators import sf_component
from signalflow.strategy.component.base import EntryRule, ExitRule
from signalflow.strategy.runner.base import StrategyRunner

_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


@dataclass
@sf_component(name="realtime_runner")
class RealtimeRunner(StrategyRunner):
    """Async paper/live trading runner.

    Polls a raw data store for new bars, computes signals via a
    ``SignalDetector``, and executes the same per-bar pipeline as
    ``BacktestRunner``.  State is checkpointed after every bar for
    crash recovery.

    For paper trading, pair with ``VirtualRealtimeBroker`` (recommended)
    or ``BacktestBroker`` + ``VirtualSpotExecutor``.

    Attributes:
        strategy_id: Unique identifier for this strategy run.
        pairs: Trading pairs to monitor.
        timeframe: Candle interval (must be a key in ``_TIMEFRAME_MINUTES``).
        initial_capital: Starting cash balance for a fresh run.
        poll_interval_sec: Seconds between store polls when no new bars.
        warmup_bars: Historical bars to load for feature warmup.
        summary_interval: Log a full summary every *N* bars.
        detector: Signal detector (with its ``FeaturePipeline``).
        broker: Order execution broker.
        raw_store: Store to read OHLCV data from.
        strategy_store: Store for state persistence.
        loader: Optional ``BinanceSpotLoader`` for built-in sync.
        sync_interval_sec: Interval for the loader sync task.
        alert_manager: Optional ``AlertManager`` for monitoring alerts.
    """

    strategy_id: str = "realtime"
    pairs: list[str] = field(default_factory=list)
    timeframe: str = "1m"
    initial_capital: float = 10000.0
    poll_interval_sec: float = 5.0
    warmup_bars: int = 100
    summary_interval: int = 10

    # Components
    detector: Any = None  # SignalDetector
    broker: Any = None
    raw_store: Any = None  # RawDataStore
    strategy_store: Any = None  # StrategyStore
    entry_rules: list[EntryRule] = field(default_factory=list)
    exit_rules: list[ExitRule] = field(default_factory=list)
    metrics: list[StrategyMetric] = field(default_factory=list)

    # Optional data sync
    loader: Any = None  # BinanceSpotLoader | None
    sync_interval_sec: int = 60

    # Optional monitoring
    alert_manager: Any = None  # AlertManager | None

    # Column config
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    price_col: str = "close"
    data_key: str = "spot"

    # Internal state
    _shutdown: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _bars_processed: int = field(default=0, init=False, repr=False)
    _trades: list[Trade] = field(default_factory=list, init=False, repr=False)
    _metrics_history: list[dict] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        raw_data: RawData | None = None,
        signals: Signals | None = None,
        state: StrategyState | None = None,
    ) -> StrategyState:
        """Sync wrapper - delegates to :meth:`run_async`.

        ``raw_data`` and ``signals`` are ignored (the runner generates
        its own).  Provided for ``StrategyRunner`` interface compat.
        """
        return asyncio.run(self.run_async(state=state))

    async def run_async(self, *, state: StrategyState | None = None) -> StrategyState:
        """Run the realtime trading loop.

        Args:
            state: Optional initial state.  When ``None`` the runner
                   tries to restore from ``strategy_store``; if nothing
                   is found a fresh state is created.

        Returns:
            Final ``StrategyState`` after shutdown.
        """
        state = self._init_state(state)

        self._install_signal_handlers()

        sync_task: asyncio.Task | None = None
        if self.loader is not None:
            sync_task = asyncio.create_task(self.loader.sync(self.pairs, update_interval_sec=self.sync_interval_sec))
            logger.info(f"Started background data sync pairs={self.pairs} interval={self.sync_interval_sec}s")

        logger.info(
            f"RealtimeRunner started strategy_id={self.strategy_id} "
            f"pairs={self.pairs} tf={self.timeframe} "
            f"poll={self.poll_interval_sec}s warmup={self.warmup_bars}"
        )

        try:
            state = await self._main_loop(state)
        finally:
            # Graceful cleanup
            if sync_task is not None:
                sync_task.cancel()
                try:
                    await sync_task
                except asyncio.CancelledError:
                    pass

            self._persist_cycle(state, trades=[], ts=state.last_ts)
            n_open = len(state.portfolio.open_positions()) if hasattr(state.portfolio, "open_positions") else 0
            logger.info(
                f"RealtimeRunner stopped - bars={self._bars_processed} "
                f"trades={len(self._trades)} "
                f"open_positions={n_open}"
            )

        return state

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    async def _main_loop(self, state: StrategyState) -> StrategyState:
        while not self._shutdown.is_set():
            new_timestamps = self._poll_new_bars(state)

            if not new_timestamps:
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(),
                        timeout=self.poll_interval_sec,
                    )
                except TimeoutError:
                    pass
                continue

            for ts in new_timestamps:
                if self._shutdown.is_set():
                    break

                bar_df, bar_signals = self._detect_signals(ts)
                if bar_df.height == 0:
                    logger.debug(f"No bar data for ts={ts}, skipping")
                    continue

                cycle_trades: list[Trade] = []
                state = self._process_bar(ts, bar_df, bar_signals, state, cycle_trades)
                self._bars_processed += 1

                self._persist_cycle(state, cycle_trades, ts)
                self._log_bar_summary(ts, state, bar_df, cycle_trades, bar_signals)

                if self.alert_manager is not None:
                    self.alert_manager.check_all(state, bar_signals, ts)

        return state

    # ------------------------------------------------------------------ #
    #  Per-bar processing (adapted from BacktestRunner._process_bar)      #
    # ------------------------------------------------------------------ #

    def _process_bar(
        self,
        ts: datetime,
        bar_df: pl.DataFrame,
        signals: Signals,
        state: StrategyState,
        out_trades: list[Trade],
    ) -> StrategyState:
        """Process a single bar - identical logic to BacktestRunner."""
        state.touch(ts)
        state.reset_tick_cache()
        state.runtime["_bar_signals"] = signals

        prices = self._build_prices(bar_df)
        self.broker.mark_positions(state, prices, ts)

        # Metrics
        all_metrics: dict[str, float] = {"timestamp": ts.timestamp()}
        for metric in self.metrics:
            metric_values = metric.compute(state, prices)
            all_metrics.update(metric_values)
        state.metrics = all_metrics
        self._metrics_history.append(all_metrics.copy())

        # Exits
        exit_orders = []
        open_positions = state.portfolio.open_positions()
        for exit_rule in self.exit_rules:
            orders = exit_rule.check_exits(open_positions, prices, state)
            exit_orders.extend(orders)

        if exit_orders:
            exit_fills = self.broker.submit_orders(exit_orders, prices, ts)
            exit_trades = self.broker.process_fills(exit_fills, exit_orders, state)
            self._trades.extend(exit_trades)
            out_trades.extend(exit_trades)

        # Entries
        entry_orders = []
        for entry_rule in self.entry_rules:
            orders = entry_rule.check_entries(signals, prices, state)
            entry_orders.extend(orders)

        if entry_orders:
            entry_fills = self.broker.submit_orders(entry_orders, prices, ts)
            entry_trades = self.broker.process_fills(entry_fills, entry_orders, state)
            self._trades.extend(entry_trades)
            out_trades.extend(entry_trades)

        return state

    # ------------------------------------------------------------------ #
    #  Signal detection                                                   #
    # ------------------------------------------------------------------ #

    def _detect_signals(self, ts: datetime) -> tuple[pl.DataFrame, Signals]:
        """Load warmup window and run the detector for *ts*.

        Returns:
            (bar_df, signals) where bar_df contains only rows at *ts*
            and signals are filtered to *ts*.
        """
        tf_minutes = _TIMEFRAME_MINUTES.get(self.timeframe, 1)
        warmup_start = ts - timedelta(minutes=tf_minutes * self.warmup_bars)

        window_df = self.raw_store.load_many(
            self.pairs,
            start=warmup_start,
            end=ts,
        )

        if window_df.height == 0:
            return pl.DataFrame(), Signals(pl.DataFrame())

        raw_data = RawData(
            datetime_start=warmup_start,
            datetime_end=ts,
            pairs=self.pairs,
            data={self.data_key: window_df},
        )
        view = RawDataView(raw=raw_data)

        try:
            signals = self.detector.run(view)
        except Exception:
            logger.exception(f"Detector failed at ts={ts}")
            signals = Signals(pl.DataFrame())

        # Filter to current bar
        bar_df = window_df.filter(pl.col(self.ts_col) == ts)

        signals_df = signals.value
        if signals_df.height > 0 and self.ts_col in signals_df.columns:
            signals_df = signals_df.filter(pl.col(self.ts_col) == ts)
        bar_signals = Signals(signals_df)

        return bar_df, bar_signals

    # ------------------------------------------------------------------ #
    #  Polling                                                            #
    # ------------------------------------------------------------------ #

    def _poll_new_bars(self, state: StrategyState) -> list[datetime]:
        """Return sorted timestamps newer than ``state.last_ts``."""
        if not self.pairs:
            return []

        # Find the latest timestamp across all pairs
        latest: datetime | None = None
        for pair in self.pairs:
            _, pair_max = self.raw_store.get_time_bounds(pair)
            if pair_max is not None:
                if latest is None or pair_max > latest:
                    latest = pair_max

        if latest is None:
            return []

        if state.last_ts is not None and latest <= state.last_ts:
            return []

        # Load bars newer than last_ts (or all available when starting fresh)
        if state.last_ts is not None:
            start = state.last_ts
        else:
            # First run - find earliest timestamp across all pairs
            earliest: datetime | None = None
            for pair in self.pairs:
                pair_min, _ = self.raw_store.get_time_bounds(pair)
                if pair_min is not None:
                    if earliest is None or pair_min < earliest:
                        earliest = pair_min
            start = earliest if earliest is not None else latest
        df = self.raw_store.load_many(self.pairs, start=start, end=latest)

        if df.height == 0:
            return []

        timestamps = df.select(self.ts_col).unique().sort(self.ts_col).get_column(self.ts_col).to_list()

        # Filter out already-processed bar
        if state.last_ts is not None:
            timestamps = [t for t in timestamps if t > state.last_ts]

        self._detect_gaps(timestamps)
        return timestamps

    def _detect_gaps(self, timestamps: list[datetime]) -> None:
        """Log warnings for missing bars in the timestamp sequence."""
        if len(timestamps) < 2:
            return

        expected_delta = timedelta(minutes=_TIMEFRAME_MINUTES.get(self.timeframe, 1))

        for i in range(1, len(timestamps)):
            actual_delta = timestamps[i] - timestamps[i - 1]
            if actual_delta > expected_delta:
                missing_count = int(actual_delta / expected_delta) - 1
                logger.warning(
                    f"Gap detected: {missing_count} missing bar(s) "
                    f"between {timestamps[i - 1]} and {timestamps[i]} "
                    f"(expected={expected_delta}, actual={actual_delta})"
                )

    # ------------------------------------------------------------------ #
    #  State management                                                   #
    # ------------------------------------------------------------------ #

    def _init_state(self, state: StrategyState | None) -> StrategyState:
        if state is not None:
            return state

        if self.strategy_store is not None:
            restored = self.strategy_store.load_state(self.strategy_id)
            if restored is not None:
                state = self._reconstruct_state(restored)
                logger.info(f"Restored state strategy_id={self.strategy_id} last_ts={state.last_ts}")
                return state

        new_state = StrategyState(strategy_id=self.strategy_id)
        new_state.portfolio.cash = self.initial_capital
        return new_state

    @staticmethod
    def _reconstruct_state(raw: StrategyState) -> StrategyState:
        """Rebuild a proper ``StrategyState`` from a naively deserialized one.

        ``state_from_json`` does ``StrategyState(**data)`` which leaves
        ``portfolio`` as a plain dict and ``last_ts`` as a string.  This
        helper converts them back to the correct types.
        """
        from signalflow.core.containers.portfolio import Portfolio
        from signalflow.core.containers.position import Position

        state = StrategyState(strategy_id=raw.strategy_id)

        # last_ts: str → datetime
        if isinstance(raw.last_ts, str):
            state.last_ts = datetime.fromisoformat(raw.last_ts)
        else:
            state.last_ts = raw.last_ts

        state.last_event_id = raw.last_event_id

        # portfolio: dict → Portfolio
        if isinstance(raw.portfolio, dict):
            state.portfolio = Portfolio(
                cash=float(raw.portfolio.get("cash", 0.0)),
            )
            raw_positions = raw.portfolio.get("positions", {})
            for pos_id, pos_data in raw_positions.items():
                if isinstance(pos_data, dict):
                    # Convert datetime strings in position data
                    for dt_field in ("entry_time", "last_time"):
                        val = pos_data.get(dt_field)
                        if isinstance(val, str):
                            pos_data[dt_field] = datetime.fromisoformat(val)
                    state.portfolio.positions[pos_id] = Position(**pos_data)
                else:
                    state.portfolio.positions[pos_id] = pos_data
        else:
            state.portfolio = raw.portfolio

        # Copy remaining fields
        if isinstance(raw.runtime, dict):
            state.runtime = raw.runtime
        if isinstance(raw.metrics, dict):
            state.metrics = raw.metrics
        if isinstance(raw.metrics_phase_done, set):
            state.metrics_phase_done = raw.metrics_phase_done

        return state

    def _persist_cycle(
        self,
        state: StrategyState,
        trades: list[Trade],
        ts: datetime | None,
    ) -> None:
        if self.strategy_store is None:
            return

        try:
            self.strategy_store.save_state(state)

            for trade in trades:
                self.strategy_store.append_trade(self.strategy_id, trade)

            if ts is not None and state.metrics:
                self.strategy_store.append_metrics(
                    self.strategy_id,
                    ts,
                    state.metrics,
                )
                if hasattr(state.portfolio, "open_positions"):
                    self.strategy_store.upsert_positions(
                        self.strategy_id,
                        ts,
                        state.portfolio.open_positions(),
                    )
        except Exception:
            logger.exception("Failed to persist state")

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def _log_bar_summary(
        self,
        ts: datetime,
        state: StrategyState,
        bar_df: pl.DataFrame,
        trades: list[Trade],
        signals: Signals,
    ) -> None:
        equity = state.metrics.get("equity", 0.0)
        cash = state.portfolio.cash if hasattr(state.portfolio, "cash") else 0.0
        n_open = len(state.portfolio.open_positions()) if hasattr(state.portfolio, "open_positions") else 0
        n_signals = signals.value.height if signals.value.height > 0 else 0
        n_trades = len(trades)

        logger.info(
            f"bar ts={ts} equity={equity:.2f} cash={cash:.2f} positions={n_open} signals={n_signals} trades={n_trades}"
        )

        if self.summary_interval > 0 and self._bars_processed % self.summary_interval == 0:
            m = state.metrics
            logger.info(
                f"--- summary (bar #{self._bars_processed}) --- "
                f"total_return={m.get('total_return', 0.0):.4f} "
                f"max_drawdown={m.get('max_drawdown', 0.0):.4f} "
                f"win_rate={m.get('win_rate', 0.0):.4f} "
                f"sharpe={m.get('sharpe_ratio', 0.0):.4f}"
            )

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_prices(self, bar_data: pl.DataFrame) -> dict[str, float]:
        prices: dict[str, float] = {}
        for row in bar_data.iter_rows(named=True):
            pair = row.get(self.pair_col)
            price = row.get(self.price_col)
            if pair and price is not None:
                prices[pair] = float(price)
        return prices

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Windows or no running loop - signals handled elsewhere
            pass

    def _request_shutdown(self) -> None:
        logger.info("Shutdown signal received - finishing current bar")
        self._shutdown.set()

    @property
    def trades(self) -> list[Trade]:
        return self._trades

    @property
    def metrics_df(self) -> pl.DataFrame:
        if not self._metrics_history:
            return pl.DataFrame()
        return pl.DataFrame(self._metrics_history)
