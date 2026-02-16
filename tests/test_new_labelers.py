"""Tests for new labelers: VolatilityRegime, TrendScanning, Structure, VolumeRegime."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.target.structure_labeler import StructureLabeler
from signalflow.target.trend_scanning import TrendScanningLabeler
from signalflow.target.volatility_labeler import VolatilityRegimeLabeler
from signalflow.target.volume_labeler import VolumeRegimeLabeler

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_df(
    close: list[float],
    pair: str = "BTCUSDT",
    volume: list[float] | None = None,
) -> pl.DataFrame:
    """Build an OHLCV DataFrame from a close price series."""
    n = len(close)
    base = datetime(2024, 1, 1)
    if volume is None:
        volume = [1000.0] * n
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": volume,
        }
    )


def _sine_prices(n: int = 500, amplitude: float = 20.0, period: int = 100, base_price: float = 100.0) -> list[float]:
    """Generate sine-wave close prices."""
    return [base_price + amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def _linear_prices(n: int = 200, start: float = 100.0, slope: float = 1.0) -> list[float]:
    """Generate linearly trending close prices."""
    return [start + slope * i for i in range(n)]


def _step_prices(n: int = 300, low: float = 100.0, high: float = 200.0, step_at: int = 150) -> list[float]:
    """Generate step-function close prices."""
    return [low if i < step_at else high for i in range(n)]


# ── VolatilityRegimeLabeler ───────────────────────────────────────────────


class TestVolatilityRegimeLabeler:
    """Tests for VolatilityRegimeLabeler."""

    def test_high_vol_gets_high_volatility(self):
        """A period of wild swings should produce 'high_volatility' labels."""
        n = 600
        base = 100.0
        close = []
        for i in range(n):
            if 100 <= i <= 200:
                # High volatility zone: large random-like swings
                close.append(base + 30.0 * math.sin(i * 0.5))
            else:
                # Calm zone: tiny moves
                close.append(base + 0.01 * (i % 3))

        df = _make_df(close)
        labeler = VolatilityRegimeLabeler(
            horizon=20,
            lookback_window=300,
            upper_quantile=0.8,
            lower_quantile=0.2,
            mask_to_signals=False,
            include_meta=True,
        )
        result = labeler.compute(df)

        assert result.height == n
        # Bars just before the volatile period (fwd vol captures it) should be high_volatility
        high_volatility = result.filter(pl.col("label") == "high_volatility")
        assert high_volatility.height > 0, "Expected at least some high_volatility labels"
        assert "realized_vol" in result.columns
        assert "vol_percentile" in result.columns

    def test_calm_period_gets_low_volatility(self):
        """A flat series in a mixed dataset should produce 'low_volatility' labels."""
        n = 600
        close = []
        for i in range(n):
            if i < 200:
                # Very volatile
                close.append(100.0 + 20.0 * math.sin(i * 0.7))
            else:
                # Dead calm
                close.append(100.0 + 0.001 * i)

        df = _make_df(close)
        labeler = VolatilityRegimeLabeler(
            horizon=20,
            lookback_window=400,
            upper_quantile=0.75,
            lower_quantile=0.25,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        low_volatility = result.filter(pl.col("label") == "low_volatility")
        assert low_volatility.height > 0, "Expected at least some low_volatility labels"

    def test_length_preserving(self):
        """Output must have the same row count as input."""
        close = _sine_prices(300)
        df = _make_df(close)
        labeler = VolatilityRegimeLabeler(horizon=20, lookback_window=100, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height

    def test_validation_bad_quantiles(self):
        """lower_quantile >= upper_quantile should raise."""
        with pytest.raises(ValueError, match="lower_quantile"):
            VolatilityRegimeLabeler(lower_quantile=0.8, upper_quantile=0.2)


# ── TrendScanningLabeler ─────────────────────────────────────────────────


class TestTrendScanningLabeler:
    """Tests for TrendScanningLabeler."""

    def test_strong_uptrend_gets_rise(self):
        """A strongly rising price series should produce 'rise' labels."""
        # Add small noise to avoid zero residuals (perfect fit => MSE=0 => t-stat undefined)
        np.random.seed(42)
        close = [100.0 + 2.0 * i + np.random.normal(0, 0.1) for i in range(200)]
        df = _make_df(close)
        labeler = TrendScanningLabeler(
            min_lookforward=5,
            max_lookforward=30,
            step=5,
            critical_value=1.96,
            mask_to_signals=False,
            include_meta=True,
        )
        result = labeler.compute(df)

        rise = result.filter(pl.col("label") == "rise")
        assert rise.height > 0, "Expected 'rise' labels for a strong uptrend"
        assert "t_stat" in result.columns
        assert "best_window" in result.columns

        # No 'fall' labels for a pure uptrend
        fall = result.filter(pl.col("label") == "fall")
        assert fall.height == 0, "Should not see 'fall' in pure uptrend"

    def test_strong_downtrend_gets_fall(self):
        """A strongly falling price series should produce 'fall' labels."""
        # Add small noise to avoid zero residuals (perfect fit => MSE=0 => t-stat undefined)
        np.random.seed(42)
        close = [500.0 - 2.0 * i + np.random.normal(0, 0.1) for i in range(200)]
        df = _make_df(close)
        labeler = TrendScanningLabeler(
            min_lookforward=5,
            max_lookforward=30,
            step=5,
            critical_value=1.96,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        fall = result.filter(pl.col("label") == "fall")
        assert fall.height > 0, "Expected 'fall' labels for a strong downtrend"

        rise = result.filter(pl.col("label") == "rise")
        assert rise.height == 0, "Should not see 'rise' in pure downtrend"

    def test_flat_market_gets_null(self):
        """A perfectly flat price should yield no significant labels."""
        close = [100.0] * 200
        df = _make_df(close)
        labeler = TrendScanningLabeler(
            min_lookforward=5,
            max_lookforward=30,
            step=5,
            critical_value=1.96,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        labeled = result.filter(pl.col("label").is_not_null())
        assert labeled.height == 0, "Flat market should produce no labels"

    def test_length_preserving(self):
        """Output must have the same row count as input."""
        close = _sine_prices(200)
        df = _make_df(close)
        labeler = TrendScanningLabeler(
            min_lookforward=5,
            max_lookforward=30,
            step=5,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert result.height == df.height

    def test_validation_min_lookforward(self):
        """min_lookforward < 3 should raise."""
        with pytest.raises(ValueError, match="min_lookforward"):
            TrendScanningLabeler(min_lookforward=2)


# ── StructureLabeler ──────────────────────────────────────────────────────


class TestStructureLabeler:
    """Tests for StructureLabeler."""

    def test_peak_gets_local_max(self):
        """The peak of a sine wave should be labeled 'local_max'."""
        # sine with period 100 => peak at i=25 (sin(pi/2)=1)
        close = _sine_prices(n=200, amplitude=30.0, period=100, base_price=100.0)
        df = _make_df(close)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_pct=0.01,
            mask_to_signals=False,
            include_meta=True,
        )
        result = labeler.compute(df)

        tops = result.filter(pl.col("label") == "local_max")
        assert tops.height > 0, "Expected at least one local_max at sine peak"
        assert "swing_pct" in result.columns

    def test_trough_gets_local_min(self):
        """The trough of a sine wave should be labeled 'local_min'."""
        close = _sine_prices(n=200, amplitude=30.0, period=100, base_price=100.0)
        df = _make_df(close)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_pct=0.01,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        bottoms = result.filter(pl.col("label") == "local_min")
        assert bottoms.height > 0, "Expected at least one local_min at sine trough"

    def test_multi_pair(self):
        """Structure labeler should work independently per pair."""
        close1 = _sine_prices(n=200, amplitude=30.0, period=100, base_price=100.0)
        close2 = _sine_prices(n=200, amplitude=30.0, period=100, base_price=200.0)
        df1 = _make_df(close1, pair="BTCUSDT")
        df2 = _make_df(close2, pair="ETHUSDT")
        df = pl.concat([df1, df2])

        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_pct=0.01,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        assert result.height == 400
        btc_tops = result.filter((pl.col("pair") == "BTCUSDT") & (pl.col("label") == "local_max"))
        eth_tops = result.filter((pl.col("pair") == "ETHUSDT") & (pl.col("label") == "local_max"))
        assert btc_tops.height > 0
        assert eth_tops.height > 0

    def test_min_swing_filter(self):
        """Labels should be filtered out when swing < min_swing_pct."""
        # Tiny amplitude sine: swing ~ 0.2% which is < 2%
        close = _sine_prices(n=200, amplitude=0.1, period=100, base_price=100.0)
        df = _make_df(close)
        labeler = StructureLabeler(
            lookforward=30,
            lookback=30,
            min_swing_pct=0.02,  # 2% minimum
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        labeled = result.filter(pl.col("label").is_not_null())
        assert labeled.height == 0, "Tiny swings should be filtered by min_swing_pct"

    def test_length_preserving(self):
        """Output must have the same row count as input."""
        close = _sine_prices(200)
        df = _make_df(close)
        labeler = StructureLabeler(lookforward=30, lookback=30, min_swing_pct=0.01, mask_to_signals=False)
        result = labeler.compute(df)
        assert result.height == df.height


# ── VolumeRegimeLabeler ───────────────────────────────────────────────────


class TestVolumeRegimeLabeler:
    """Tests for VolumeRegimeLabeler."""

    def test_high_volume_gets_spike(self):
        """A sudden volume spike should produce 'abnormal_volume' labels."""
        n = 400
        volume = [100.0] * n
        # Inject a spike in the forward window of bars around index 150
        for i in range(160, 200):
            volume[i] = 5000.0

        close = [100.0 + 0.01 * i for i in range(n)]
        df = _make_df(close, volume=volume)

        labeler = VolumeRegimeLabeler(
            horizon=30,
            vol_sma_window=100,
            spike_threshold=2.0,
            drought_threshold=0.3,
            mask_to_signals=False,
            include_meta=True,
        )
        result = labeler.compute(df)

        spikes = result.filter(pl.col("label") == "abnormal_volume")
        assert spikes.height > 0, "Expected 'abnormal_volume' labels during spike period"
        assert "volume_ratio" in result.columns

    def test_low_volume_gets_drought(self):
        """A sudden volume drop should produce 'illiquidity' labels."""
        n = 400
        volume = [1000.0] * n
        # Inject drought in the forward window of bars around index 150
        for i in range(160, 250):
            volume[i] = 10.0  # Very low volume

        close = [100.0 + 0.01 * i for i in range(n)]
        df = _make_df(close, volume=volume)

        labeler = VolumeRegimeLabeler(
            horizon=30,
            vol_sma_window=100,
            spike_threshold=2.0,
            drought_threshold=0.3,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        droughts = result.filter(pl.col("label") == "illiquidity")
        assert droughts.height > 0, "Expected 'illiquidity' labels during drought period"

    def test_length_preserving(self):
        """Output must have the same row count as input."""
        close = [100.0 + 0.01 * i for i in range(300)]
        volume = [1000.0] * 300
        df = _make_df(close, volume=volume)

        labeler = VolumeRegimeLabeler(
            horizon=20,
            vol_sma_window=100,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert result.height == df.height

    def test_validation_bad_thresholds(self):
        """drought_threshold >= spike_threshold should raise."""
        with pytest.raises(ValueError, match="drought_threshold"):
            VolumeRegimeLabeler(spike_threshold=1.0, drought_threshold=2.0)
