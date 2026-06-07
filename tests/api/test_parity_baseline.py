"""Parity baseline — golden anchor for the v2 refactor (REFACTOR_PLAN Етап 0).

These tests pin the END-TO-END signal-resolution result on deterministic data so
that later refactor stages (especially removing ``.features()``/``_feature_pipelines``
from ``FlowBuilder``) must prove "результат не змінився". A change here is allowed
ONLY as a conscious update of the golden value.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.api.flow import flow
from signalflow.core import RawData


@pytest.fixture
def deterministic_raw() -> RawData:
    """300 deterministic 1m bars for one pair (no randomness, no I/O)."""
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(300):
        price = 50000.0 + 500 * math.sin(i / 12.0) + i * 5
        rows.append(
            {
                "pair": "BTCUSDT",
                "timestamp": base + timedelta(minutes=i),
                "open": price - 5,
                "high": price + 20,
                "low": price - 20,
                "close": price,
                "volume": 1000.0 + i,
            }
        )
    df = pl.DataFrame(rows)
    return RawData(
        datetime_start=base,
        datetime_end=base + timedelta(minutes=300),
        pairs=["BTCUSDT"],
        data={"spot": df},
    )


class TestParityBaseline:
    """Golden anchors that any flow refactor must keep stable."""

    def test_signal_count_and_distribution(self, deterministic_raw: RawData):
        """example/sma_cross on the fixed series yields exactly 6 signals (3 rise / 3 fall)."""
        builder = flow().data(raw=deterministic_raw).detector("example/sma_cross")
        signals, _detector_features = builder.resolve_signals(deterministic_raw)
        v = signals.value

        assert v.height == 6, f"signal count drifted: {v.height} (expected 6)"
        by_type = {row["signal_type"]: row["len"] for row in v.group_by("signal_type").len().to_dicts()}
        assert by_type == {"rise": 3, "fall": 3}, f"distribution drifted: {by_type}"

    def test_signal_resolution_is_deterministic(self, deterministic_raw: RawData):
        """Two independent resolutions of the same flow are bit-identical."""
        b1 = flow().data(raw=deterministic_raw).detector("example/sma_cross")
        b2 = flow().data(raw=deterministic_raw).detector("example/sma_cross")
        s1, _ = b1.resolve_signals(deterministic_raw)
        s2, _ = b2.resolve_signals(deterministic_raw)
        assert s1.value.equals(s2.value)

    def test_signals_schema(self, deterministic_raw: RawData):
        """Resolved signals carry the canonical columns."""
        builder = flow().data(raw=deterministic_raw).detector("example/sma_cross")
        signals, _ = builder.resolve_signals(deterministic_raw)
        for col in ("pair", "timestamp", "signal_type", "signal"):
            assert col in signals.value.columns
