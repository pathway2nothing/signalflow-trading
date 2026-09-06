"""Opt-in performance guards for the backtest loop: ``SF_BENCH=1 pytest tests/test_bench.py``."""

import os
import time

import pytest

import signalflow as sf

pytestmark = pytest.mark.skipif(not os.environ.get("SF_BENCH"), reason="set SF_BENCH=1 to run the benchmarks")


def test_backtest_loop_357k_bars_under_10s():
    ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2025-01-01", end="2025-09-06", interval="1m")
    flow = sf.Flow(name="bench", detectors=[sf.SmaCrossDetector(fast=10, slow=30)], strategy=sf.RulesStrategy())
    t0 = time.perf_counter()
    run = flow.backtest(ds, capital=10_000)
    elapsed = time.perf_counter() - t0
    assert run.equity_curve.height == ds.height + 1
    assert elapsed < 10.0, f"backtest over {ds.height:,} bars took {elapsed:.1f}s"


def test_backtest_two_pairs_two_weeks_under_1_5s():
    ds = sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2025-01-01", end="2025-01-15", interval="1m")
    flow = sf.Flow(name="bench2", detectors=[sf.SmaCrossDetector(fast=10, slow=30)], strategy=sf.RulesStrategy())
    t0 = time.perf_counter()
    flow.backtest(ds, capital=10_000)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"took {elapsed:.2f}s"
