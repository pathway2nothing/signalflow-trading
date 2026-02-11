"""Tests for StructureLabeler and ZigzagStructureLabeler."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core.enums import SignalCategory
from signalflow.target.structure_labeler import StructureLabeler, ZigzagStructureLabeler


def _sine_wave_df(n=500, pair="BTCUSDT", period=100, amplitude=10.0):
    """OHLCV with a sine wave pattern (clear tops and bottoms)."""
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    prices = [100.0 + amplitude * np.sin(2 * np.pi * i / period) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _flat_df(n=200, pair="BTCUSDT"):
    """OHLCV with constant price (no structure)."""
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": [100.0] * n,
            "high": [100.1] * n,
            "low": [99.9] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )


def _two_regime_df(n=1000, pair="BTCUSDT"):
    """Low-vol segment then high-vol segment (for z-score testing)."""
    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]
    prices = []
    # Low vol first half: amplitude=1
    for i in range(n // 2):
        prices.append(100.0 + 1.0 * np.sin(2 * np.pi * i / 50))
    # High vol second half: amplitude=20
    for i in range(n // 2):
        prices.append(100.0 + 20.0 * np.sin(2 * np.pi * i / 50))
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


# ── StructureLabeler (window-based) ─────────────────────────────────────


class TestStructureLabeler:
    def test_signal_category(self):
        labeler = StructureLabeler()
        assert labeler.signal_category == SignalCategory.PRICE_STRUCTURE

    def test_output_length_preserved(self):
        df = _sine_wave_df(300)
        labeler = StructureLabeler(lookforward=30, lookback=30, min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_sine_wave_has_tops_and_bottoms(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        labeler = StructureLabeler(lookforward=30, lookback=30, min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        labels = result["label"].to_list()
        has_top = any(l == "local_max" for l in labels)
        has_bottom = any(l == "local_min" for l in labels)
        assert has_top, "Sine wave should have local tops"
        assert has_bottom, "Sine wave should have local bottoms"

    def test_flat_market_no_labels(self):
        df = _flat_df(200)
        labeler = StructureLabeler(lookforward=30, lookback=30, min_swing_pct=0.02, mask_to_signals=False)
        result = labeler.compute(df)
        labels = result["label"].to_list()
        non_null = [l for l in labels if l is not None]
        assert len(non_null) == 0, "Flat market should have no structure labels"

    def test_labels_are_valid_values(self):
        df = _sine_wave_df(300)
        labeler = StructureLabeler(lookforward=30, lookback=30, mask_to_signals=False)
        result = labeler.compute(df)
        valid = {"local_max", "local_min", None}
        unique = set(result["label"].to_list())
        assert unique <= valid

    def test_meta_columns(self):
        df = _sine_wave_df(300)
        labeler = StructureLabeler(lookforward=30, lookback=30, include_meta=True, mask_to_signals=False)
        result = labeler.compute(df)
        assert "swing_pct" in result.columns

    def test_invalid_lookback(self):
        with pytest.raises(ValueError, match="lookback"):
            StructureLabeler(lookback=0)

    def test_invalid_lookforward(self):
        with pytest.raises(ValueError, match="lookforward"):
            StructureLabeler(lookforward=0)

    def test_multi_pair(self):
        btc = _sine_wave_df(200, pair="BTCUSDT")
        eth = _sine_wave_df(200, pair="ETHUSDT")
        df = pl.concat([btc, eth])
        labeler = StructureLabeler(lookforward=20, lookback=20, min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height
        pairs = result["pair"].unique().to_list()
        assert set(pairs) == {"BTCUSDT", "ETHUSDT"}


class TestStructureLabelerZScore:
    def test_zscore_produces_labels(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_zscore=1.0,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        has_top = any(l == "local_max" for l in labels)
        has_bottom = any(l == "local_min" for l in labels)
        assert has_top or has_bottom, "Z-score mode should detect extrema in sine wave"

    def test_zscore_flat_no_labels(self):
        df = _flat_df(300)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_zscore=2.0,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        non_null = [l for l in labels if l is not None]
        assert len(non_null) == 0

    def test_zscore_adapts_to_volatility(self):
        """Z-score mode should find extrema in both low-vol and high-vol segments."""
        df = _two_regime_df(1000)
        labeler = StructureLabeler(
            lookforward=15,
            lookback=15,
            min_swing_zscore=1.5,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        low_vol = result.head(500).filter(pl.col("label").is_not_null()).height
        high_vol = result.tail(500).filter(pl.col("label").is_not_null()).height

        assert low_vol > 0, "Should detect extrema in low vol segment"
        assert high_vol > 0, "Should detect extrema in high vol segment"

    def test_zscore_invalid_params(self):
        with pytest.raises(ValueError, match="min_swing_zscore"):
            StructureLabeler(min_swing_zscore=0)

        with pytest.raises(ValueError, match="min_swing_zscore"):
            StructureLabeler(min_swing_zscore=-1.0)

        with pytest.raises(ValueError, match="vol_window"):
            StructureLabeler(min_swing_zscore=2.0, vol_window=5)

    def test_zscore_meta_columns(self):
        df = _sine_wave_df(500)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_zscore=1.0,
            vol_window=100,
            include_meta=True,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert "swing_pct" in result.columns

    def test_zscore_length_preserved(self):
        df = _sine_wave_df(500)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_zscore=2.0,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert result.height == df.height


# ── ZigzagStructureLabeler (global) ─────────────────────────────────────


class TestZigzagStructureLabeler:
    def test_signal_category(self):
        labeler = ZigzagStructureLabeler()
        assert labeler.signal_category == SignalCategory.PRICE_STRUCTURE

    def test_length_preserved(self):
        df = _sine_wave_df(500)
        labeler = ZigzagStructureLabeler(min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_sine_wave_has_alternating_pivots(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        labeler = ZigzagStructureLabeler(min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        labels = result["label"].to_list()

        # Extract non-null labels in order
        pivots = [(i, l) for i, l in enumerate(labels) if l is not None]
        assert len(pivots) >= 2, "Should find at least 2 pivots"

        has_top = any(l == "local_max" for _, l in pivots)
        has_bottom = any(l == "local_min" for _, l in pivots)
        assert has_top and has_bottom

        # Verify strict alternation
        for j in range(1, len(pivots)):
            assert pivots[j][1] != pivots[j - 1][1], (
                f"Pivots must alternate: pivot {j - 1}={pivots[j - 1][1]} "
                f"at idx {pivots[j - 1][0]}, pivot {j}={pivots[j][1]} at idx {pivots[j][0]}"
            )

    def test_flat_market_no_labels(self):
        df = _flat_df(300)
        labeler = ZigzagStructureLabeler(min_swing_pct=0.02, mask_to_signals=False)
        result = labeler.compute(df)
        labels = result["label"].to_list()
        non_null = [l for l in labels if l is not None]
        assert len(non_null) == 0

    def test_valid_label_values(self):
        df = _sine_wave_df(300)
        labeler = ZigzagStructureLabeler(min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        valid = {"local_max", "local_min", None}
        unique = set(result["label"].to_list())
        assert unique <= valid

    def test_meta_columns(self):
        df = _sine_wave_df(300)
        labeler = ZigzagStructureLabeler(min_swing_pct=0.01, include_meta=True, mask_to_signals=False)
        result = labeler.compute(df)
        assert "swing_pct" in result.columns
        # swing_pct should be positive for all pivots
        pivots = result.filter(pl.col("label").is_not_null())
        assert pivots.filter(pl.col("swing_pct") > 0).height == pivots.height

    def test_multi_pair(self):
        btc = _sine_wave_df(300, pair="BTCUSDT")
        eth = _sine_wave_df(300, pair="ETHUSDT")
        df = pl.concat([btc, eth])
        labeler = ZigzagStructureLabeler(min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height
        pairs = result["pair"].unique().to_list()
        assert set(pairs) == {"BTCUSDT", "ETHUSDT"}

    def test_invalid_swing_pct(self):
        with pytest.raises(ValueError, match="min_swing_pct"):
            ZigzagStructureLabeler(min_swing_pct=0)

        with pytest.raises(ValueError, match="min_swing_pct"):
            ZigzagStructureLabeler(min_swing_pct=-0.01)

    def test_higher_threshold_fewer_pivots(self):
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        results = {}
        for pct in [0.01, 0.05, 0.15]:
            labeler = ZigzagStructureLabeler(min_swing_pct=pct, mask_to_signals=False)
            result = labeler.compute(df)
            n_pivots = result.filter(pl.col("label").is_not_null()).height
            results[pct] = n_pivots

        # Higher threshold -> fewer pivots
        assert results[0.01] >= results[0.05] >= results[0.15]

    def test_tops_at_highs_bottoms_at_lows(self):
        """Verify that tops are at high prices and bottoms at low prices."""
        df = _sine_wave_df(500, period=100, amplitude=10.0)
        labeler = ZigzagStructureLabeler(
            min_swing_pct=0.01,
            include_meta=True,
            keep_input_columns=True,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        tops = result.filter(pl.col("label") == "local_max")
        bottoms = result.filter(pl.col("label") == "local_min")

        if tops.height > 0 and bottoms.height > 0:
            avg_top = tops["close"].mean()
            avg_bottom = bottoms["close"].mean()
            assert avg_top > avg_bottom, "Tops should be at higher prices than bottoms"


class TestZigzagZScore:
    def test_zscore_produces_labels(self):
        df = _sine_wave_df(600, period=100, amplitude=10.0)
        labeler = ZigzagStructureLabeler(
            min_swing_zscore=1.5,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        has_top = any(l == "local_max" for l in labels)
        has_bottom = any(l == "local_min" for l in labels)
        assert has_top or has_bottom

    def test_zscore_alternation(self):
        df = _sine_wave_df(600, period=100, amplitude=10.0)
        labeler = ZigzagStructureLabeler(
            min_swing_zscore=1.0,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        pivots = [l for l in labels if l is not None]

        for j in range(1, len(pivots)):
            assert pivots[j] != pivots[j - 1], "Pivots must alternate in z-score mode"

    def test_zscore_flat_no_labels(self):
        df = _flat_df(300)
        labeler = ZigzagStructureLabeler(
            min_swing_zscore=2.0,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        labels = result["label"].to_list()
        non_null = [l for l in labels if l is not None]
        assert len(non_null) == 0

    def test_zscore_adapts_to_volatility(self):
        """Z-score mode should find pivots in both regimes."""
        df = _two_regime_df(1000)
        labeler = ZigzagStructureLabeler(
            min_swing_zscore=1.5,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        low_vol = result.head(500).filter(pl.col("label").is_not_null()).height
        high_vol = result.tail(500).filter(pl.col("label").is_not_null()).height

        assert low_vol > 0, "Should detect pivots in low vol segment"
        assert high_vol > 0, "Should detect pivots in high vol segment"

    def test_zscore_invalid_params(self):
        with pytest.raises(ValueError, match="min_swing_zscore"):
            ZigzagStructureLabeler(min_swing_zscore=0)

        with pytest.raises(ValueError, match="vol_window"):
            ZigzagStructureLabeler(min_swing_zscore=2.0, vol_window=5)

    def test_zscore_length_preserved(self):
        df = _sine_wave_df(500)
        labeler = ZigzagStructureLabeler(
            min_swing_zscore=2.0,
            vol_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert result.height == df.height
