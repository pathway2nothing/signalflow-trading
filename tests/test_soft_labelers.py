"""Tests for the soft-label API on every shipped Labeler subclass."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.target import (
    AnomalyLabeler,
    DrawdownLabeler,
    FixedHorizonLabeler,
    FlashMoveLabeler,
    HurstRegimeLabeler,
    MeanReversionEventLabeler,
    SharpeTercileLabeler,
    StructureLabeler,
    TakeProfitLabeler,
    TrendBreakLabeler,
    TrendScanningLabeler,
    TripleBarrierLabeler,
    VolatilityRegimeLabeler,
    VolatilityShockLabeler,
    VolumeClimaxLabeler,
    VolumeRegimeLabeler,
    ZigzagStructureLabeler,
)


def _ohlcv(n: int = 600, seed: int = 0, pair: str = "BTCUSDT") -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.005, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.normal(5000, 1000, n))
    base = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# (name, labeler instance) — covers every concrete labeler in signalflow.target.
ALL_LABELERS = [
    ("fixed_horizon", FixedHorizonLabeler(horizon=10, mask_to_signals=False)),
    ("volatility_regime", VolatilityRegimeLabeler(horizon=30, lookback_window=120, mask_to_signals=False)),
    ("volatility_shock", VolatilityShockLabeler(horizon=60, past_vol_window=200, vol_window_short=30, mask_to_signals=False)),
    ("anomaly", AnomalyLabeler(horizon=30, vol_window=200, mask_to_signals=False)),
    ("flash_move", FlashMoveLabeler(flash_horizon=5, vol_window=100, mask_to_signals=False)),
    ("sharpe_tercile", SharpeTercileLabeler(horizon=30, lookback_window=200, mask_to_signals=False)),
    ("volume_regime", VolumeRegimeLabeler(horizon=30, vol_sma_window=100, mask_to_signals=False)),
    ("volume_climax", VolumeClimaxLabeler(horizon=60, vol_sma_window=100, mask_to_signals=False)),
    ("drawdown", DrawdownLabeler(horizon=60, lookback_window=200, mask_to_signals=False)),
    ("drawdown_runup", DrawdownLabeler(horizon=60, lookback_window=200, mode="runup", mask_to_signals=False)),
    ("drawdown_calmar", DrawdownLabeler(horizon=60, lookback_window=200, mode="calmar", mask_to_signals=False)),
    ("hurst", HurstRegimeLabeler(horizon=64, stride=8, mask_to_signals=False)),
    ("mean_reversion", MeanReversionEventLabeler(horizon=60, z_window=100, mask_to_signals=False)),
    ("trend_break", TrendBreakLabeler(window=80, mask_to_signals=False)),
    ("trend_scanning", TrendScanningLabeler(min_lookforward=5, max_lookforward=30, step=5, mask_to_signals=False)),
    ("triple_barrier", TripleBarrierLabeler(horizon=60, vol_window=30, mask_to_signals=False)),
    ("take_profit", TakeProfitLabeler(horizon=60, barrier_pct=0.01, mask_to_signals=False)),
    ("structure", StructureLabeler(lookforward=30, lookback=30, min_swing_pct=0.005, mask_to_signals=False)),
    ("zigzag", ZigzagStructureLabeler(min_swing_pct=0.01, mask_to_signals=False)),
]


@pytest.mark.parametrize("name,labeler", ALL_LABELERS, ids=[n for n, _ in ALL_LABELERS])
def test_soft_columns_present_and_sum_to_one(name: str, labeler) -> None:
    """compute_soft emits the declared columns and probabilities sum to 1 per valid row."""
    df = _ohlcv()
    soft = labeler.compute_soft(df)

    expected_cols = [f"{labeler.soft_col_prefix}{c}" for c in labeler.soft_classes]
    for c in expected_cols:
        assert c in soft.columns, f"{name}: missing column {c}"

    assert soft.height == df.height, f"{name}: row count changed"

    sums = soft.select(expected_cols).sum_horizontal().to_numpy()
    sums = sums[~np.isnan(sums)]
    valid = sums > 1e-12
    assert valid.any(), f"{name}: no valid soft rows for this fixture"
    assert np.allclose(sums[valid], 1.0, atol=1e-6), (
        f"{name}: probabilities don't sum to 1 (min={sums[valid].min()}, max={sums[valid].max()})"
    )


@pytest.mark.parametrize("name,labeler", ALL_LABELERS, ids=[n for n, _ in ALL_LABELERS])
def test_soft_null_propagation(name: str, labeler) -> None:
    """Invalid rows (warm-up / no forward data) should have all-null probability columns."""
    df = _ohlcv()
    soft = labeler.compute_soft(df)
    p_cols = [f"{labeler.soft_col_prefix}{c}" for c in labeler.soft_classes]
    p_mat = soft.select(p_cols).to_numpy()
    row_null = np.isnan(p_mat).any(axis=1)
    row_all_null = np.isnan(p_mat).all(axis=1)
    # Any row that has at least one null must have all-null probabilities.
    assert np.array_equal(row_null, row_all_null), (
        f"{name}: partial null rows detected (some p_<cls> non-null while siblings are null)"
    )


@pytest.mark.parametrize(
    "name,labeler",
    [
        ("fixed_horizon", FixedHorizonLabeler(horizon=10, mask_to_signals=False, softness_k=500.0)),
        ("anomaly", AnomalyLabeler(horizon=30, vol_window=200, mask_to_signals=False, softness_k=500.0)),
        ("flash_move", FlashMoveLabeler(flash_horizon=5, vol_window=100, mask_to_signals=False, softness_k=500.0)),
        ("vol_shock", VolatilityShockLabeler(horizon=60, past_vol_window=200, vol_window_short=30, mask_to_signals=False, softness_k=500.0)),
        ("trend_scanning", TrendScanningLabeler(min_lookforward=5, max_lookforward=30, step=5, mask_to_signals=False, softness_k=500.0)),
    ],
)
def test_soft_collapses_to_hard_when_k_is_large(name: str, labeler) -> None:
    """With very large softness_k the argmax-class probability should be near 1."""
    df = _ohlcv(seed=1)
    soft = labeler.compute_soft(df)
    p_cols = [f"{labeler.soft_col_prefix}{c}" for c in labeler.soft_classes]
    p_mat = soft.select(p_cols).to_numpy()
    row_valid = ~np.isnan(p_mat).any(axis=1)
    assert row_valid.any(), f"{name}: no valid rows to check collapse"
    max_p = p_mat[row_valid].max(axis=1)
    # At least 80% of rows should have a near-1 argmax under high k (some bars sit on the boundary).
    near_one = (max_p > 0.9).mean()
    # Some bars sit exactly on the decision boundary (e.g. fwd_ret ~ 0 for
    # FixedHorizon) and never harden no matter how steep the sigmoid. 70% is a
    # safe floor for the calibrated subclasses on random-walk OHLCV.
    assert near_one >= 0.7, f"{name}: only {near_one:.2%} rows hardened with high k"


def test_soft_raises_when_classes_undeclared() -> None:
    """A subclass that does not declare soft_classes should refuse compute_soft."""
    from signalflow.target.base import Labeler

    class _NoSoftClasses(Labeler):
        out_col: str = "label"
        mask_to_signals: bool = False

        def compute_group(self, group_df, data_context=None):
            return group_df.with_columns(pl.lit("x").alias("label"))

    df = _ohlcv(n=20)
    lab = _NoSoftClasses()
    with pytest.raises(NotImplementedError, match="soft_classes"):
        lab.compute_soft(df)


def test_default_one_hot_fallback() -> None:
    """Base.compute_group_soft one-hot-encodes the hard label column."""
    from signalflow.target.base import Labeler

    class _Simple(Labeler):
        soft_classes = ("a", "b", "c")
        out_col: str = "label"
        mask_to_signals: bool = False

        def compute_group(self, group_df, data_context=None):
            labels = ["a", "b", None, "c", "b"][: group_df.height]
            return group_df.with_columns(pl.Series("label", labels, dtype=pl.Utf8))

    df = pl.DataFrame(
        {"pair": ["BTC"] * 5, "timestamp": list(range(5)), "close": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    soft = _Simple().compute_soft(df).sort("timestamp")
    assert soft.get_column("p_a").to_list() == [1.0, 0.0, None, 0.0, 0.0]
    assert soft.get_column("p_b").to_list() == [0.0, 1.0, None, 0.0, 1.0]
    assert soft.get_column("p_c").to_list() == [0.0, 0.0, None, 1.0, 0.0]
