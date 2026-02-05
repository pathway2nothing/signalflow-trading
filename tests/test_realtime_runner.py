"""Tests for RealtimeRunner using VirtualDataProvider."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import SignalType
from signalflow.data.raw_store import DuckDbSpotStore
from signalflow.data.source.virtual import (
    VirtualDataProvider,
    generate_crossover_data,
    generate_ohlcv,
)
from signalflow.data.strategy_store import DuckDbStrategyStore
from signalflow.detector.sma_cross import ExampleSmaCrossDetector
from signalflow.strategy.broker.backtest import BacktestBroker
from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
from signalflow.strategy.component.entry.fixed_size import FixedSizeEntryRule
from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
from signalflow.strategy.runner.realtime_runner import RealtimeRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

START = datetime(2024, 1, 1)
PAIR = "BTCUSDT"
PAIRS = ["BTCUSDT"]


@pytest.fixture
def raw_db(tmp_path: Path) -> DuckDbSpotStore:
    store = DuckDbSpotStore(db_path=tmp_path / "raw.duckdb", timeframe="1m")
    yield store
    store.close()


@pytest.fixture
def strategy_db(tmp_path: Path) -> DuckDbStrategyStore:
    store = DuckDbStrategyStore(str(tmp_path / "strategy.duckdb"))
    store.init()
    yield store
    store.close()


@pytest.fixture
def virtual_provider(raw_db: DuckDbSpotStore) -> VirtualDataProvider:
    return VirtualDataProvider(store=raw_db, timeframe="1m", seed=42)


@pytest.fixture
def detector() -> ExampleSmaCrossDetector:
    return ExampleSmaCrossDetector(fast_period=5, slow_period=10)


@pytest.fixture
def broker(strategy_db: DuckDbStrategyStore) -> BacktestBroker:
    return BacktestBroker(
        executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
        store=strategy_db,
    )


@pytest.fixture
def runner(
    raw_db: DuckDbSpotStore,
    strategy_db: DuckDbStrategyStore,
    detector: ExampleSmaCrossDetector,
    broker: BacktestBroker,
) -> RealtimeRunner:
    return RealtimeRunner(
        strategy_id="test_rt",
        pairs=PAIRS,
        timeframe="1m",
        initial_capital=10000.0,
        poll_interval_sec=0.01,  # fast polling for tests
        warmup_bars=50,
        summary_interval=5,
        detector=detector,
        broker=broker,
        raw_store=raw_db,
        strategy_store=strategy_db,
        entry_rules=[FixedSizeEntryRule(position_size=0.1, max_positions=5)],
        exit_rules=[TakeProfitStopLossExit(take_profit_pct=0.05, stop_loss_pct=0.03)],
        metrics=[],
    )


# ---------------------------------------------------------------------------
# VirtualDataProvider tests
# ---------------------------------------------------------------------------


class TestVirtualDataProvider:
    def test_generate_ohlcv_basic(self) -> None:
        bars = generate_ohlcv(PAIR, START, n_bars=100, seed=42)
        assert len(bars) == 100
        assert bars[0]["timestamp"] == START
        assert bars[-1]["timestamp"] == START + timedelta(minutes=99)
        for bar in bars:
            assert bar["high"] >= bar["low"]
            assert bar["high"] >= bar["close"]
            assert bar["low"] <= bar["close"]
            assert bar["volume"] > 0

    def test_generate_ohlcv_reproducible(self) -> None:
        bars_a = generate_ohlcv(PAIR, START, n_bars=50, seed=123)
        bars_b = generate_ohlcv(PAIR, START, n_bars=50, seed=123)
        assert bars_a == bars_b

    def test_generate_crossover_data(self) -> None:
        bars = generate_crossover_data(PAIR, START, n_bars=100, seed=42)
        assert len(bars) == 100
        # After the reversal, the end should be higher than the middle
        mid = len(bars) // 2
        assert bars[-1]["close"] > bars[mid]["close"]

    def test_download(self, raw_db: DuckDbSpotStore) -> None:
        provider = VirtualDataProvider(store=raw_db, timeframe="1m", seed=42)
        provider.download(pairs=PAIRS, n_bars=100, start=START)

        df = raw_db.load(PAIR)
        assert df.height == 100
        assert df["timestamp"].min() == START

    def test_download_multiple_pairs(self, raw_db: DuckDbSpotStore) -> None:
        pairs = ["BTCUSDT", "ETHUSDT"]
        provider = VirtualDataProvider(store=raw_db, timeframe="1m", seed=42)
        provider.download(pairs=pairs, n_bars=50, start=START)

        for pair in pairs:
            df = raw_db.load(pair)
            assert df.height == 50

    @pytest.mark.asyncio
    async def test_sync(self, raw_db: DuckDbSpotStore) -> None:
        provider = VirtualDataProvider(store=raw_db, timeframe="1m", seed=42)
        # Pre-populate
        provider.download(pairs=PAIRS, n_bars=10, start=START)

        # Run sync for a short time
        task = asyncio.create_task(provider.sync(PAIRS, update_interval_sec=0.01))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        df = raw_db.load(PAIR)
        # Should have more bars than the initial 10
        assert df.height > 10


# ---------------------------------------------------------------------------
# RealtimeRunner unit tests
# ---------------------------------------------------------------------------


class TestPollNewBars:
    def test_returns_empty_when_no_data(self, runner: RealtimeRunner) -> None:
        state = StrategyState(strategy_id="test_rt")
        assert runner._poll_new_bars(state) == []

    def test_returns_all_bars_when_no_last_ts(
        self, runner: RealtimeRunner, virtual_provider: VirtualDataProvider
    ) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=5, start=START)
        state = StrategyState(strategy_id="test_rt")
        timestamps = runner._poll_new_bars(state)
        assert len(timestamps) == 5

    def test_returns_only_new_bars(self, runner: RealtimeRunner, virtual_provider: VirtualDataProvider) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=10, start=START)
        state = StrategyState(strategy_id="test_rt")
        state.last_ts = START + timedelta(minutes=4)

        timestamps = runner._poll_new_bars(state)
        assert len(timestamps) == 5
        assert all(t > state.last_ts for t in timestamps)

    def test_returns_empty_when_caught_up(self, runner: RealtimeRunner, virtual_provider: VirtualDataProvider) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=5, start=START)
        state = StrategyState(strategy_id="test_rt")
        state.last_ts = START + timedelta(minutes=4)

        timestamps = runner._poll_new_bars(state)
        assert timestamps == []


class TestDetectSignals:
    def test_returns_empty_when_no_data(self, runner: RealtimeRunner) -> None:
        bar_df, signals = runner._detect_signals(START)
        assert bar_df.height == 0
        assert signals.value.height == 0

    def test_returns_signals_for_timestamp(self, runner: RealtimeRunner, raw_db: DuckDbSpotStore) -> None:
        # Generate crossover data with enough warmup
        bars = generate_crossover_data(PAIR, START, n_bars=100, crossover_at=70, seed=42)
        raw_db.insert_klines(PAIR, bars)

        target_ts = START + timedelta(minutes=99)
        bar_df, signals = runner._detect_signals(target_ts)

        assert bar_df.height > 0
        # bar_df should only contain rows at target_ts
        assert bar_df["timestamp"].to_list() == [target_ts] * bar_df.height

    def test_detector_exception_returns_empty_signals(
        self, runner: RealtimeRunner, virtual_provider: VirtualDataProvider
    ) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=5, start=START)

        # Patch detector to raise
        with patch.object(runner.detector, "run", side_effect=RuntimeError("boom")):
            ts = START + timedelta(minutes=4)
            bar_df, signals = runner._detect_signals(ts)
            assert signals.value.height == 0


class TestProcessBar:
    def test_updates_state(self, runner: RealtimeRunner, virtual_provider: VirtualDataProvider) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=5, start=START)

        state = StrategyState(strategy_id="test_rt")
        state.portfolio.cash = 10000.0

        raw_df = runner.raw_store.load_many(PAIRS, start=START, end=START)
        bar_df = raw_df.filter(pl.col("timestamp") == START)
        signals = Signals(pl.DataFrame())
        trades: list = []

        state = runner._process_bar(START, bar_df, signals, state, trades)

        assert state.last_ts == START

    def test_creates_entry_on_signal(self, runner: RealtimeRunner, virtual_provider: VirtualDataProvider) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=5, start=START)

        state = StrategyState(strategy_id="test_rt")
        state.portfolio.cash = 10000.0

        raw_df = runner.raw_store.load(PAIR, start=START, end=START)

        # Create a RISE signal
        signals_df = pl.DataFrame(
            {
                "pair": [PAIR],
                "timestamp": [START],
                "signal_type": [SignalType.RISE.value],
                "signal": [1],
            }
        )
        signals = Signals(signals_df)
        trades: list = []

        state = runner._process_bar(START, raw_df, signals, state, trades)

        assert len(trades) == 1
        assert trades[0].side == "BUY"
        assert len(state.portfolio.open_positions()) == 1
        assert state.portfolio.cash < 10000.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullCycle:
    @pytest.mark.asyncio
    async def test_processes_bars_and_persists(
        self,
        runner: RealtimeRunner,
        raw_db: DuckDbSpotStore,
        strategy_db: DuckDbStrategyStore,
    ) -> None:
        # Generate crossover data for signals
        bars = generate_crossover_data(PAIR, START, n_bars=80, crossover_at=55, seed=42)
        raw_db.insert_klines(PAIR, bars)

        # Run with a shutdown after processing
        async def stop_after_delay():
            await asyncio.sleep(0.5)
            runner._request_shutdown()

        asyncio.create_task(stop_after_delay())
        state = await runner.run_async()

        # Should have processed bars
        assert runner._bars_processed > 0
        assert state.last_ts is not None
        assert state.strategy_id == "test_rt"

        # State should be persisted
        restored = strategy_db.load_state("test_rt")
        assert restored is not None
        # load_state returns last_ts as string from JSON
        restored_ts = restored.last_ts
        if isinstance(restored_ts, str):
            restored_ts = datetime.fromisoformat(restored_ts)
        assert restored_ts == state.last_ts

    @pytest.mark.asyncio
    async def test_with_virtual_sync(
        self,
        raw_db: DuckDbSpotStore,
        strategy_db: DuckDbStrategyStore,
        detector: ExampleSmaCrossDetector,
        broker: BacktestBroker,
    ) -> None:
        provider = VirtualDataProvider(store=raw_db, timeframe="1m", seed=42)
        # Pre-populate enough data for warmup
        provider.download(pairs=PAIRS, n_bars=60, start=START)

        runner = RealtimeRunner(
            strategy_id="test_sync",
            pairs=PAIRS,
            timeframe="1m",
            initial_capital=10000.0,
            poll_interval_sec=0.05,
            warmup_bars=50,
            summary_interval=0,
            detector=detector,
            broker=broker,
            raw_store=raw_db,
            strategy_store=strategy_db,
            entry_rules=[FixedSizeEntryRule(position_size=0.1, max_positions=5)],
            exit_rules=[TakeProfitStopLossExit(take_profit_pct=0.05, stop_loss_pct=0.03)],
            metrics=[],
            loader=provider,
            sync_interval_sec=0.05,
        )

        async def stop_after_delay():
            await asyncio.sleep(1.0)
            runner._request_shutdown()

        asyncio.create_task(stop_after_delay())
        state = await runner.run_async()

        assert runner._bars_processed > 0
        assert state.last_ts is not None


class TestRestartRecovery:
    @pytest.mark.asyncio
    async def test_resumes_from_persisted_state(
        self,
        raw_db: DuckDbSpotStore,
        strategy_db: DuckDbStrategyStore,
        detector: ExampleSmaCrossDetector,
    ) -> None:
        def make_runner() -> RealtimeRunner:
            return RealtimeRunner(
                strategy_id="test_recovery",
                pairs=PAIRS,
                timeframe="1m",
                initial_capital=10000.0,
                poll_interval_sec=0.01,
                warmup_bars=20,
                summary_interval=0,
                detector=detector,
                broker=BacktestBroker(
                    executor=VirtualSpotExecutor(fee_rate=0.001),
                    store=strategy_db,
                ),
                raw_store=raw_db,
                strategy_store=strategy_db,
                entry_rules=[FixedSizeEntryRule(position_size=0.1, max_positions=5)],
                exit_rules=[],
                metrics=[],
            )

        # First run — load initial batch, process, then stop
        bars_batch1 = generate_ohlcv(PAIR, START, n_bars=50, seed=42)
        raw_db.insert_klines(PAIR, bars_batch1)

        runner1 = make_runner()

        async def stop_early():
            await asyncio.sleep(0.3)
            runner1._request_shutdown()

        asyncio.create_task(stop_early())
        state1 = await runner1.run_async()
        bars_first_run = runner1._bars_processed

        assert bars_first_run > 0
        assert state1.last_ts is not None

        # Add more data AFTER first run to simulate new bars arriving
        bars_batch2 = generate_ohlcv(
            PAIR,
            START + timedelta(minutes=50),
            n_bars=50,
            base_price=bars_batch1[-1]["close"],
            seed=99,
        )
        raw_db.insert_klines(PAIR, bars_batch2)

        # Second run — should resume from last_ts and process new bars
        runner2 = make_runner()

        async def stop_again():
            await asyncio.sleep(0.3)
            runner2._request_shutdown()

        asyncio.create_task(stop_again())
        state2 = await runner2.run_async()

        # Runner2 should have processed the new bars
        assert runner2._bars_processed > 0
        assert state2.last_ts > state1.last_ts


class TestGracefulShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_event(
        self,
        runner: RealtimeRunner,
        virtual_provider: VirtualDataProvider,
        strategy_db: DuckDbStrategyStore,
    ) -> None:
        virtual_provider.download(pairs=PAIRS, n_bars=60, start=START)

        async def send_shutdown():
            await asyncio.sleep(0.2)
            runner._request_shutdown()

        asyncio.create_task(send_shutdown())
        state = await runner.run_async()

        # Runner should have exited cleanly
        assert runner._shutdown.is_set()
        # State should be saved
        restored = strategy_db.load_state("test_rt")
        assert restored is not None


class TestNoNewBars:
    @pytest.mark.asyncio
    async def test_sleeps_when_no_bars(self, runner: RealtimeRunner) -> None:
        """Runner should sleep and not crash when store is empty."""

        async def stop_quick():
            await asyncio.sleep(0.1)
            runner._request_shutdown()

        asyncio.create_task(stop_quick())
        state = await runner.run_async()

        assert runner._bars_processed == 0
        assert state.portfolio.cash == 10000.0
