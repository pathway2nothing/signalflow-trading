"""Tests for AnomalyLabeler and AnomalyDetector.

Uses synthetic data: normal random walk with injected extreme returns
to test extreme positive anomaly / extreme negative anomaly detection.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.core import RawData, RawDataView
from signalflow.core.enums import SignalCategory
from signalflow.detector.anomaly_detector import AnomalyDetector
from signalflow.target.anomaly_labeler import AnomalyLabeler

# ── Helpers ────────────────────────────────────────────────────────────────


def _random_walk_df(
    n: int = 3000,
    pair: str = "BTCUSDT",
    base_price: float = 100.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a random-walk OHLCV DataFrame with small normal returns.

    Returns a DataFrame with ``n`` rows of synthetic 1-minute candles.
    The returns are drawn from N(0, 0.001) so that a 5-sigma event has
    magnitude ~0.005 which is well within "normal" territory.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.001, size=n)
    log_returns[0] = 0.0  # first bar has no return
    log_prices = np.log(base_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    base_ts = datetime(2024, 1, 1)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(n)]

    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": timestamps,
            "open": prices.tolist(),
            "high": (prices * 1.001).tolist(),
            "low": (prices * 0.999).tolist(),
            "close": prices.tolist(),
            "volume": [1000.0] * n,
        }
    )


def _inject_spike(
    df: pl.DataFrame,
    index: int,
    magnitude: float,
) -> pl.DataFrame:
    """Inject a single-bar price spike at ``index``.

    ``magnitude`` is a multiplicative factor applied to close at ``index``.
    A magnitude of 1.05 means a +5% jump, 0.90 means a -10% drop.
    """
    close_col = df["close"].to_list()
    open_col = df["open"].to_list()
    high_col = df["high"].to_list()
    low_col = df["low"].to_list()

    new_price = close_col[index - 1] * magnitude
    close_col[index] = new_price
    open_col[index] = new_price
    high_col[index] = new_price * 1.001
    low_col[index] = new_price * 0.999

    # Propagate forward so the series stays continuous
    for i in range(index + 1, len(close_col)):
        ratio = close_col[i] / df["close"][i]
        factor = new_price / df["close"][index]
        close_col[i] = df["close"][i] * factor
        open_col[i] = df["open"][i] * factor
        high_col[i] = df["high"][i] * factor
        low_col[i] = df["low"][i] * factor

    return df.with_columns(
        [
            pl.Series("close", close_col),
            pl.Series("open", open_col),
            pl.Series("high", high_col),
            pl.Series("low", low_col),
        ]
    )


def _inject_forward_spike(
    df: pl.DataFrame,
    start_index: int,
    magnitude: float,
    over_bars: int = 1,
) -> pl.DataFrame:
    """Inject a spike that happens ``over_bars`` bars after ``start_index``.

    This creates a sharp price movement starting at ``start_index + 1``
    and completing at ``start_index + over_bars``, so that the labeler
    sees a large forward return when looking from ``start_index``.
    """
    close_col = df["close"].to_list()
    open_col = df["open"].to_list()
    high_col = df["high"].to_list()
    low_col = df["low"].to_list()

    base_price = close_col[start_index]
    target_price = base_price * magnitude

    # Linear interpolation to target over `over_bars` bars
    for step in range(1, over_bars + 1):
        idx = start_index + step
        if idx >= len(close_col):
            break
        frac = step / over_bars
        new_price = base_price + (target_price - base_price) * frac
        close_col[idx] = new_price
        open_col[idx] = new_price
        high_col[idx] = new_price * 1.001
        low_col[idx] = new_price * 0.999

    # Propagate forward from end of spike
    end_idx = min(start_index + over_bars, len(close_col) - 1)
    factor = close_col[end_idx] / df["close"][end_idx]
    for i in range(end_idx + 1, len(close_col)):
        close_col[i] = df["close"][i] * factor
        open_col[i] = df["open"][i] * factor
        high_col[i] = df["high"][i] * factor
        low_col[i] = df["low"][i] * factor

    return df.with_columns(
        [
            pl.Series("close", close_col),
            pl.Series("open", open_col),
            pl.Series("high", high_col),
            pl.Series("low", low_col),
        ]
    )


# ── AnomalyLabeler Tests ──────────────────────────────────────────────────


class TestAnomalyLabelerNormalMarket:
    """Test that normal market data produces null labels."""

    def test_normal_returns_null_labels(self):
        """Normal random walk should produce no anomaly labels."""
        df = _random_walk_df(n=3000, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        non_null = result.filter(pl.col("label").is_not_null())
        # With a 4-sigma threshold on a normal random walk,
        # anomalies should be extremely rare or absent
        assert non_null.height == 0, f"Expected no anomaly labels in normal market, got {non_null.height}"


class TestAnomalyLabelerExtremePositive:
    """Test that extreme positive spikes get labeled as extreme_positive_anomaly."""

    def test_large_positive_spike_labeled(self):
        """A 5-sigma positive forward return should be labeled extreme_positive_anomaly."""
        df = _random_walk_df(n=3000, seed=42)
        # Inject a massive upward spike: +15% over 60 bars
        # With vol ~0.001 per bar, 60-bar vol ~0.001*sqrt(60) ~0.0077
        # A 15% move is log(1.15) ~0.14, which is ~18 sigma
        df = _inject_forward_spike(df, start_index=2000, magnitude=1.15, over_bars=60)

        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        # The bar at index 2000 should have a label
        ts_at_2000 = df["timestamp"][2000]
        row = result.filter(pl.col("timestamp") == ts_at_2000)
        assert row.height == 1
        label_val = row["label"][0]
        assert label_val == "extreme_positive_anomaly", f"Expected 'extreme_positive_anomaly', got '{label_val}'"


class TestAnomalyLabelerExtremeNegative:
    """Test that sharp negative spikes get labeled as extreme_negative_anomaly."""

    def test_sharp_negative_spike_labeled(self):
        """A large negative return within flash_horizon should be extreme_negative_anomaly."""
        df = _random_walk_df(n=3000, seed=42)
        # Inject a crash: -15% happening within 5 bars (< flash_horizon=10)
        df = _inject_forward_spike(df, start_index=2000, magnitude=0.85, over_bars=5)

        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            flash_horizon=10,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        ts_at_2000 = df["timestamp"][2000]
        row = result.filter(pl.col("timestamp") == ts_at_2000)
        assert row.height == 1
        label_val = row["label"][0]
        assert label_val == "extreme_negative_anomaly", f"Expected 'extreme_negative_anomaly', got '{label_val}'"


class TestAnomalyLabelerOutputSchema:
    """Test output schema conformance."""

    def test_output_columns_default(self):
        """Default output should have pair, timestamp, label."""
        df = _random_walk_df(n=500, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=100,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert "pair" in result.columns
        assert "timestamp" in result.columns
        assert "label" in result.columns
        # OHLCV columns should NOT be in output
        assert "close" not in result.columns
        assert "open" not in result.columns

    def test_label_dtype_is_utf8(self):
        """Label column should be Utf8 (string) dtype."""
        df = _random_walk_df(n=500, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=100,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert result.schema["label"] == pl.Utf8


class TestAnomalyLabelerLengthPreserving:
    """Test that output row count matches input."""

    def test_length_preserved(self):
        """Output must have the same number of rows as input."""
        df = _random_walk_df(n=2000, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert result.height == df.height


class TestAnomalyLabelerMultiPair:
    """Test multi-pair behavior."""

    def test_multi_pair_independent_labeling(self):
        """Each pair should be labeled independently."""
        btc = _random_walk_df(n=3000, pair="BTCUSDT", seed=42)
        eth = _random_walk_df(n=3000, pair="ETHUSDT", seed=99)

        # Inject spike only in BTC
        btc = _inject_forward_spike(btc, start_index=2000, magnitude=1.20, over_bars=60)

        df = pl.concat([btc, eth])

        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)

        # Total row count preserved
        assert result.height == df.height

        # BTC should have at least one anomaly label
        btc_labels = result.filter((pl.col("pair") == "BTCUSDT") & pl.col("label").is_not_null())
        assert btc_labels.height > 0, "BTC should have anomaly labels"

        # ETH should have no anomaly labels (normal market)
        eth_labels = result.filter((pl.col("pair") == "ETHUSDT") & pl.col("label").is_not_null())
        assert eth_labels.height == 0, f"ETH should have no anomaly labels, got {eth_labels.height}"


class TestAnomalyLabelerMetaColumns:
    """Test include_meta=True produces metadata columns."""

    def test_meta_columns_present(self):
        """When include_meta=True, forward_ret and vol columns should appear."""
        df = _random_walk_df(n=2000, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            include_meta=True,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert "forward_ret" in result.columns
        assert "vol" in result.columns

    def test_meta_forward_ret_values(self):
        """forward_ret should be log(close[t+h]/close[t])."""
        df = _random_walk_df(n=2000, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            include_meta=True,
            mask_to_signals=False,
        )
        result = labeler.compute(df, data_context=None)

        # Manually check a specific row
        idx = 1000
        close_now = df["close"][idx]
        close_future = df["close"][idx + 60]
        expected_ret = math.log(close_future / close_now)

        actual_ret = result["forward_ret"][idx]
        assert actual_ret == pytest.approx(expected_ret, rel=1e-6)

    def test_meta_vol_positive(self):
        """vol column should be positive where not null."""
        df = _random_walk_df(n=2000, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=500,
            threshold_return_std=4.0,
            include_meta=True,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        vol_non_null = result.filter(pl.col("vol").is_not_null())
        assert (vol_non_null["vol"] > 0).all()

    def test_meta_not_present_by_default(self):
        """When include_meta=False (default), meta columns are absent."""
        df = _random_walk_df(n=500, seed=42)
        labeler = AnomalyLabeler(
            horizon=60,
            vol_window=100,
            threshold_return_std=4.0,
            mask_to_signals=False,
        )
        result = labeler.compute(df)
        assert "forward_ret" not in result.columns
        assert "vol" not in result.columns


class TestAnomalyLabelerValidation:
    """Test parameter validation."""

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon must be > 0"):
            AnomalyLabeler(horizon=0)

    def test_invalid_vol_window_raises(self):
        with pytest.raises(ValueError, match="vol_window must be > 0"):
            AnomalyLabeler(vol_window=0)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold_return_std must be > 0"):
            AnomalyLabeler(threshold_return_std=-1.0)

    def test_signal_category_is_anomaly(self):
        labeler = AnomalyLabeler(mask_to_signals=False)
        assert labeler.signal_category == SignalCategory.ANOMALY


# ── AnomalyDetector Tests ─────────────────────────────────────────────────


def _make_raw_data_view(df: pl.DataFrame, pairs: list[str] | None = None) -> RawDataView:
    """Helper to create a RawDataView from a DataFrame."""
    if pairs is None:
        pairs = df["pair"].unique().to_list()
    raw = RawData(
        datetime_start=df["timestamp"].min(),
        datetime_end=df["timestamp"].max(),
        pairs=pairs,
        data={"spot": df},
    )
    return RawDataView(raw=raw)


class TestAnomalyDetector:
    """Test the backward-looking anomaly detector."""

    def test_detector_finds_backward_anomaly(self):
        """Detector should find an injected single-bar spike."""
        df = _random_walk_df(n=3000, seed=42)
        # Inject a single-bar crash at bar 2000: -15%
        df = _inject_spike(df, index=2000, magnitude=0.85)

        view = _make_raw_data_view(df)
        detector = AnomalyDetector(
            vol_window=500,
            threshold_return_std=4.0,
        )
        signals = detector.run(view)

        s = signals.value
        assert s.height > 0, "Detector should find at least one anomaly"

        # The spike at bar 2000 should be detected
        ts_at_2000 = df["timestamp"][2000]
        spike_row = s.filter(pl.col("timestamp") == ts_at_2000)
        assert spike_row.height == 1, "Spike bar should be detected"
        assert spike_row["signal_type"][0] == "extreme_negative_anomaly"

    def test_detector_no_signals_in_normal_market(self):
        """Normal random walk should produce no anomaly signals."""
        df = _random_walk_df(n=3000, seed=42)
        view = _make_raw_data_view(df)

        detector = AnomalyDetector(
            vol_window=500,
            threshold_return_std=4.0,
        )
        signals = detector.run(view)

        s = signals.value
        assert s.height == 0, f"Expected no signals in normal market, got {s.height}"

    def test_detector_output_columns(self):
        """Detector output should have required columns."""
        df = _random_walk_df(n=3000, seed=42)
        df = _inject_spike(df, index=2000, magnitude=1.20)

        view = _make_raw_data_view(df)
        detector = AnomalyDetector(vol_window=500, threshold_return_std=4.0)
        signals = detector.run(view)

        s = signals.value
        assert "pair" in s.columns
        assert "timestamp" in s.columns
        assert "signal_type" in s.columns
        assert "signal" in s.columns
        assert "probability" in s.columns

    def test_detector_positive_spike_is_extreme_positive(self):
        """A positive single-bar spike should be labeled extreme_positive_anomaly."""
        df = _random_walk_df(n=3000, seed=42)
        df = _inject_spike(df, index=2000, magnitude=1.15)

        view = _make_raw_data_view(df)
        detector = AnomalyDetector(vol_window=500, threshold_return_std=4.0)
        signals = detector.run(view)

        s = signals.value
        ts_at_2000 = df["timestamp"][2000]
        spike_row = s.filter(pl.col("timestamp") == ts_at_2000)
        assert spike_row.height == 1
        assert spike_row["signal_type"][0] == "extreme_positive_anomaly"

    def test_detector_signal_category(self):
        """Detector signal_category should be ANOMALY."""
        detector = AnomalyDetector()
        assert detector.signal_category == SignalCategory.ANOMALY

    def test_detector_allowed_signal_types(self):
        """Detector should allow extreme_positive_anomaly and extreme_negative_anomaly signal types."""
        detector = AnomalyDetector()
        assert detector.allowed_signal_types == {"extreme_positive_anomaly", "extreme_negative_anomaly"}

    def test_detector_probability_between_0_and_1(self):
        """Probability values should be between 0 and 1."""
        df = _random_walk_df(n=3000, seed=42)
        df = _inject_spike(df, index=2000, magnitude=0.80)

        view = _make_raw_data_view(df)
        detector = AnomalyDetector(vol_window=500, threshold_return_std=4.0)
        signals = detector.run(view)

        s = signals.value
        probs = s.filter(pl.col("probability").is_not_null())["probability"]
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()
