"""Tests for the four revert/meta labelers added in iter-32 follow-up:

* MeanReversionMagnitudeLabeler
* DirectionalMeanReversionLabeler
* TimeToBarrierLabeler
* MetaLabelLabeler
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalflow.target import (
    DirectionalMeanReversionLabeler,
    MeanReversionMagnitudeLabeler,
    MetaLabelLabeler,
    TimeToBarrierLabeler,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_df(close: list[float], pair: str = "BTCUSDT") -> pl.DataFrame:
    n = len(close)
    base = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "pair": [pair] * n,
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "open": close,
            "high": [c + 0.1 for c in close],
            "low": [c - 0.1 for c in close],
            "close": close,
            "volume": [1000.0] * n,
        }
    )


def _spike_then_revert(
    n: int = 600,
    base: float = 100.0,
    spike_at: int = 300,
    spike_size: float = 5.0,
    revert_in: int = 50,
    noise: float = 0.01,
) -> list[float]:
    """Flat noisy series with a single up-spike that reverts toward base.

    The pre-spike portion has tight ``noise``-scale variance so the rolling
    σ stays small; the post-spike decay returns linearly toward base over
    ``revert_in`` bars, after which it stays at base.
    """
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        if i < spike_at:
            out.append(base + rng.normal(0, noise))
        elif i == spike_at:
            out.append(base + spike_size)
        elif i < spike_at + revert_in:
            frac = (i - spike_at) / revert_in
            out.append(base + spike_size * (1.0 - frac) + rng.normal(0, noise))
        else:
            out.append(base + rng.normal(0, noise))
    return out


def _spike_no_revert(
    n: int = 600,
    base: float = 100.0,
    spike_at: int = 300,
    spike_size: float = 5.0,
    noise: float = 0.01,
) -> list[float]:
    """Flat then permanent regime shift up by spike_size — no reversion."""
    rng = np.random.default_rng(11)
    out = []
    for i in range(n):
        if i < spike_at:
            out.append(base + rng.normal(0, noise))
        else:
            out.append(base + spike_size + rng.normal(0, noise))
    return out


def _down_spike_then_revert(
    n: int = 600,
    base: float = 100.0,
    spike_at: int = 300,
    spike_size: float = 5.0,
    revert_in: int = 50,
    noise: float = 0.01,
) -> list[float]:
    rng = np.random.default_rng(13)
    out = []
    for i in range(n):
        if i < spike_at:
            out.append(base + rng.normal(0, noise))
        elif i == spike_at:
            out.append(base - spike_size)
        elif i < spike_at + revert_in:
            frac = (i - spike_at) / revert_in
            out.append(base - spike_size * (1.0 - frac) + rng.normal(0, noise))
        else:
            out.append(base + rng.normal(0, noise))
    return out


# ── MeanReversionMagnitudeLabeler ──────────────────────────────────────────


class TestMeanReversionMagnitudeLabeler:
    def test_length_preserving_and_columns(self):
        df = _make_df(_spike_then_revert())
        lab = MeanReversionMagnitudeLabeler(
            horizon=80,
            z_window=120,
            include_meta=True,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        assert out.height == df.height
        for col in ("label", "z_now", "revert_strength"):
            assert col in out.columns

    def test_full_revert_detected_after_spike(self):
        close = _spike_then_revert(spike_size=8.0, revert_in=40)
        df = _make_df(close)
        lab = MeanReversionMagnitudeLabeler(
            horizon=80,
            z_window=120,
            stretch_threshold=2.0,
            partial_threshold=0.25,
            full_threshold=0.75,
            include_meta=True,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        # At least some bars around the spike should be labelled full_revert.
        full = out.filter(pl.col("label") == "full_revert")
        assert full.height > 0, "expected full_revert labels around the spike"
        # Revert strength on full_revert rows must clear the threshold.
        if "revert_strength" in full.columns:
            vals = full.get_column("revert_strength").drop_nulls()
            assert vals.min() >= 0.75 - 1e-9

    def test_no_revert_when_step_change(self):
        close = _spike_no_revert(spike_size=8.0)
        df = _make_df(close)
        lab = MeanReversionMagnitudeLabeler(
            horizon=80,
            z_window=120,
            stretch_threshold=2.0,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        # Bars immediately after the step should be "no_revert" — they stay
        # overstretched against the pre-step baseline.
        labels = out.get_column("label").to_list()
        no_revert_count = sum(1 for x in labels if x == "no_revert")
        full_count = sum(1 for x in labels if x == "full_revert")
        assert no_revert_count > 0
        assert no_revert_count > full_count

    def test_non_overstretched_bars_null(self):
        close = [100.0 + 0.001 * i for i in range(500)]
        df = _make_df(close)
        lab = MeanReversionMagnitudeLabeler(
            horizon=80,
            z_window=120,
            stretch_threshold=2.0,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        labels = out.get_column("label").to_list()
        # All bars are calm — almost everything should be null.
        non_null = [x for x in labels if x is not None]
        assert len(non_null) < 30, f"expected few labels on flat series, got {len(non_null)}"

    def test_soft_probabilities_sum_to_one(self):
        df = _make_df(_spike_then_revert())
        lab = MeanReversionMagnitudeLabeler(horizon=80, z_window=120, mask_to_signals=False)
        soft = lab.compute_soft(df)
        cols = ["p_no_revert", "p_partial_revert", "p_full_revert"]
        s = soft.select(cols).to_numpy()
        # On non-null rows the row sum must equal 1 ± tiny tolerance.
        finite = ~np.isnan(s).any(axis=1)
        sums = s[finite].sum(axis=1)
        assert finite.sum() > 0
        assert np.allclose(sums, 1.0, atol=1e-6)


# ── DirectionalMeanReversionLabeler ────────────────────────────────────────


class TestDirectionalMeanReversionLabeler:
    def test_long_revert_on_down_spike(self):
        close = _down_spike_then_revert(spike_size=8.0, revert_in=40)
        df = _make_df(close)
        lab = DirectionalMeanReversionLabeler(
            horizon=80,
            z_window=120,
            stretch_threshold=2.0,
            revert_threshold=0.5,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        labels = out.get_column("label").drop_nulls().to_list()
        n_long = labels.count("revert_long")
        n_short = labels.count("revert_short")
        assert n_long > 0
        assert n_long > n_short

    def test_short_revert_on_up_spike(self):
        close = _spike_then_revert(spike_size=8.0, revert_in=40)
        df = _make_df(close)
        lab = DirectionalMeanReversionLabeler(
            horizon=80,
            z_window=120,
            stretch_threshold=2.0,
            revert_threshold=0.5,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        labels = out.get_column("label").drop_nulls().to_list()
        n_long = labels.count("revert_long")
        n_short = labels.count("revert_short")
        assert n_short > 0
        assert n_short > n_long

    def test_no_revert_on_step(self):
        close = _spike_no_revert(spike_size=8.0)
        df = _make_df(close)
        lab = DirectionalMeanReversionLabeler(
            horizon=80,
            z_window=120,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        labels = out.get_column("label").drop_nulls().to_list()
        # Bars right after the step are overbought but z_fwd stays high → no_revert dominates.
        n_no_revert = labels.count("no_revert")
        n_short = labels.count("revert_short")
        assert n_no_revert > n_short

    def test_soft_sums_to_one(self):
        df = _make_df(_spike_then_revert())
        lab = DirectionalMeanReversionLabeler(horizon=80, z_window=120, mask_to_signals=False)
        soft = lab.compute_soft(df)
        cols = ["p_revert_short", "p_no_revert", "p_revert_long"]
        s = soft.select(cols).to_numpy()
        finite = ~np.isnan(s).any(axis=1)
        sums = s[finite].sum(axis=1)
        assert finite.sum() > 0
        assert np.allclose(sums, 1.0, atol=1e-6)


# ── TimeToBarrierLabeler ───────────────────────────────────────────────────


class TestTimeToBarrierLabeler:
    def test_emits_meta_columns(self):
        close = _spike_then_revert()
        df = _make_df(close)
        lab = TimeToBarrierLabeler(
            vol_window=30,
            horizon=60,
            profit_multiplier=1.0,
            stop_loss_multiplier=1.0,
            include_meta=True,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        for col in ("label", "hit_time", "hit_ret", "censored"):
            assert col in out.columns
        # hit_time bounded.
        ht = out.get_column("hit_time").drop_nulls().to_numpy()
        assert (ht > 0).all() and (ht <= 1.0 + 1e-9).all()

    def test_uptrend_hits_profit_barrier(self):
        # Steady uptrend should preferentially hit the upper barrier early.
        n = 600
        close = [100.0 * (1.0 + 0.0005 * i) for i in range(n)]
        df = _make_df(close)
        lab = TimeToBarrierLabeler(
            vol_window=30,
            horizon=200,
            profit_multiplier=0.5,
            stop_loss_multiplier=2.0,
            include_meta=True,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        events = out.get_column("label").drop_nulls().to_list()
        n_pt = events.count("pt")
        n_sl = events.count("sl")
        assert n_pt > n_sl

    def test_vertical_barrier_marks_censored(self):
        # Flat noise — barriers rarely touched → vertical/censored dominates.
        rng = np.random.default_rng(3)
        close = [100.0 + rng.normal(0, 0.01) for _ in range(600)]
        df = _make_df(close)
        lab = TimeToBarrierLabeler(
            vol_window=30,
            horizon=60,
            profit_multiplier=10.0,
            stop_loss_multiplier=10.0,
            include_meta=True,
            mask_to_signals=False,
        )
        out = lab.compute(df)
        events = out.get_column("label").drop_nulls().to_list()
        n_vert = events.count("vertical")
        assert n_vert > 0
        # censored flag aligns with vertical label
        ce = out.get_column("censored").to_list()
        lab_col = out.get_column("label").to_list()
        for c, lv in zip(ce, lab_col, strict=True):
            if lv == "vertical":
                assert c is True
            elif lv in ("pt", "sl"):
                assert c is False

    def test_soft_sums_to_one(self):
        df = _make_df(_spike_then_revert())
        lab = TimeToBarrierLabeler(
            vol_window=30, horizon=60, profit_multiplier=1.0, stop_loss_multiplier=1.0, mask_to_signals=False
        )
        soft = lab.compute_soft(df)
        cols = ["p_sl_fast", "p_vertical", "p_pt_fast"]
        s = soft.select(cols).to_numpy()
        finite = ~np.isnan(s).any(axis=1)
        sums = s[finite].sum(axis=1)
        assert finite.sum() > 0
        assert np.allclose(sums, 1.0, atol=1e-6)


# ── MetaLabelLabeler ───────────────────────────────────────────────────────


def _signal_keys(pair: str, df: pl.DataFrame, indices: list[int], side: int | None = None) -> pl.DataFrame:
    ts = df.get_column("timestamp").to_list()
    rows = {
        "pair": [pair] * len(indices),
        "timestamp": [ts[i] for i in indices],
    }
    if side is not None:
        rows["side"] = [side] * len(indices)
    return pl.DataFrame(rows)


class TestMetaLabelLabeler:
    def test_requires_signal_keys(self):
        df = _make_df(_spike_then_revert())
        lab = MetaLabelLabeler(horizon=20, min_return=0.005, max_loss=0.01)
        with pytest.raises(ValueError):
            lab.compute(df)

    def test_take_when_long_signal_meets_target(self):
        # Use a slowly drifting up market — long signals should resolve "take".
        n = 400
        close = [100.0 * (1.0 + 0.0008 * i) for i in range(n)]
        df = _make_df(close)
        # Place signals well before the end so the forward horizon is available.
        indices = [50, 80, 120, 160, 200]
        sk = _signal_keys("BTCUSDT", df, indices, side=1)
        lab = MetaLabelLabeler(
            horizon=80,
            mode="triple_barrier",
            min_return=0.01,
            max_loss=0.05,
            mask_to_signals=True,
            include_meta=True,
        )
        out = lab.compute(df, data_context={"signal_keys": sk})
        # Only signal bars should have labels.
        labels = out.get_column("label").to_list()
        non_null_idx = [i for i, v in enumerate(labels) if v is not None]
        assert set(non_null_idx) == set(indices)
        # All should be "take" in this drift scenario.
        non_null_vals = [labels[i] for i in non_null_idx]
        assert non_null_vals.count("take") >= 4

    def test_skip_when_short_signal_on_uptrend(self):
        n = 400
        close = [100.0 * (1.0 + 0.001 * i) for i in range(n)]
        df = _make_df(close)
        indices = [50, 80, 120, 160, 200]
        sk = _signal_keys("BTCUSDT", df, indices, side=-1)
        lab = MetaLabelLabeler(
            horizon=80,
            mode="triple_barrier",
            min_return=0.005,
            max_loss=0.02,
            mask_to_signals=True,
        )
        out = lab.compute(df, data_context={"signal_keys": sk})
        labels = out.get_column("label").to_list()
        non_null = [labels[i] for i in indices]
        # Short signals into an uptrend should mostly be "skip".
        assert non_null.count("skip") >= 4

    def test_fixed_horizon_mode_runs(self):
        n = 400
        close = [100.0 * (1.0 + 0.001 * i) for i in range(n)]
        df = _make_df(close)
        indices = [50, 80, 120]
        sk = _signal_keys("BTCUSDT", df, indices, side=1)
        lab = MetaLabelLabeler(
            horizon=80,
            mode="fixed_horizon",
            min_return=0.01,
            mask_to_signals=True,
            include_meta=True,
        )
        out = lab.compute(df, data_context={"signal_keys": sk})
        labels = out.get_column("label").to_list()
        assert all(labels[i] in ("take", "skip") for i in indices)

    def test_soft_take_probability(self):
        n = 400
        close = [100.0 * (1.0 + 0.001 * i) for i in range(n)]
        df = _make_df(close)
        indices = [50, 80, 120, 160, 200]
        sk = _signal_keys("BTCUSDT", df, indices, side=1)
        lab = MetaLabelLabeler(
            horizon=80,
            mode="triple_barrier",
            min_return=0.01,
            max_loss=0.05,
        )
        soft = lab.compute_soft(df, data_context={"signal_keys": sk})
        cols = ["p_skip", "p_take"]
        s = soft.select(cols).to_numpy()
        finite = ~np.isnan(s).any(axis=1)
        sums = s[finite].sum(axis=1)
        assert finite.sum() > 0
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_string_side_aliases(self):
        n = 400
        close = [100.0 * (1.0 - 0.001 * i) for i in range(n)]
        df = _make_df(close)
        indices = [50, 80, 120]
        ts = df.get_column("timestamp").to_list()
        sk = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * len(indices),
                "timestamp": [ts[i] for i in indices],
                "side": ["short"] * len(indices),
            }
        )
        lab = MetaLabelLabeler(
            horizon=80,
            mode="triple_barrier",
            min_return=0.005,
            max_loss=0.05,
            mask_to_signals=True,
        )
        out = lab.compute(df, data_context={"signal_keys": sk})
        labels = out.get_column("label").to_list()
        non_null = [labels[i] for i in indices]
        # Downward drift makes "short" trades "take".
        assert non_null.count("take") >= 2
