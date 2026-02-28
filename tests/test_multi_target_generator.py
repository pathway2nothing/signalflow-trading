"""Tests for MultiTargetGenerator."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow.target.fixed_horizon_labeler import FixedHorizonLabeler
from signalflow.target.multi_target_generator import (
    HorizonConfig,
    MultiTargetGenerator,
    TargetType,
)


def _make_ohlcv(n: int = 500, pairs: list[str] | None = None) -> pl.DataFrame:
    """Generate synthetic OHLCV data with a simple sine wave pattern."""
    if pairs is None:
        pairs = ["BTCUSDT", "ETHUSDT"]

    base = datetime(2024, 1, 1)
    rows = []
    for pair in pairs:
        for i in range(n):
            price = 100.0 + 10 * math.sin(i / 50.0) + i * 0.01
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(minutes=i),
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1000.0 + 500 * abs(math.sin(i / 30.0)),
                }
            )
    return pl.DataFrame(rows)


class TestMultiTargetGenerator:
    def test_generates_expected_columns_single_horizon(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="direction", kind="discrete")],
        )
        df = _make_ohlcv(200)
        result = gen.generate(df)
        assert "target_direction_short" in result.columns

    def test_preserves_row_count(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="direction", kind="discrete")],
        )
        df = _make_ohlcv(200)
        result = gen.generate(df)
        assert result.height == df.height

    def test_preserves_original_columns(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="direction", kind="discrete")],
        )
        df = _make_ohlcv(200)
        original_cols = set(df.columns)
        result = gen.generate(df)
        assert original_cols.issubset(set(result.columns))

    def test_multi_horizon_multi_target(self):
        gen = MultiTargetGenerator(
            horizons=[
                HorizonConfig(name="short", horizon=30),
                HorizonConfig(name="long", horizon=120),
            ],
            target_types=[
                TargetType(name="direction", kind="discrete"),
                TargetType(name="return_magnitude", kind="continuous"),
                TargetType(name="volume_regime", kind="discrete"),
            ],
        )
        df = _make_ohlcv(300)
        result = gen.generate(df)
        target_cols = [c for c in result.columns if c.startswith("target_")]
        # 2 horizons x 3 targets = 6 columns
        assert len(target_cols) == 6

    def test_return_magnitude_non_negative(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="return_magnitude", kind="continuous")],
        )
        df = _make_ohlcv(200)
        result = gen.generate(df)
        col = "target_return_magnitude_short"
        values = result.get_column(col).drop_nulls()
        assert (values >= 0).all()

    def test_volume_regime_values(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="volume_regime", kind="discrete")],
        )
        df = _make_ohlcv(300)
        result = gen.generate(df)
        col = "target_volume_regime_short"
        unique_vals = set(result.get_column(col).drop_nulls().to_list())
        assert unique_vals.issubset({"HIGH", "MED", "LOW"})

    def test_target_columns_metadata(self):
        gen = MultiTargetGenerator(
            horizons=[
                HorizonConfig(name="short", horizon=30),
                HorizonConfig(name="long", horizon=120),
            ],
        )
        meta = gen.target_columns()
        assert all("column" in m for m in meta)
        assert all("kind" in m for m in meta)
        assert len(meta) == 2 * len(gen.target_types)

    def test_missing_column_raises(self):
        gen = MultiTargetGenerator()
        df = pl.DataFrame({"pair": ["A"], "timestamp": [datetime(2024, 1, 1)]})
        with pytest.raises(ValueError, match="Missing required columns"):
            gen.generate(df)

    def test_empty_horizons_raises(self):
        gen = MultiTargetGenerator(horizons=[])
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match="At least one horizon"):
            gen.generate(df)

    def test_crash_regime_values(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="crash_regime", kind="discrete")],
        )
        df = _make_ohlcv(300)
        result = gen.generate(df)
        col = "target_crash_regime_short"
        assert col in result.columns
        unique_vals = set(result.get_column(col).drop_nulls().to_list())
        assert unique_vals.issubset({"crash", "rally", "normal"})

    def test_crash_regime_quantile_distribution(self):
        gen = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="crash_regime", kind="discrete")],
            crash_quantiles=(0.1, 0.9),
        )
        df = _make_ohlcv(500)
        result = gen.generate(df)
        col = "target_crash_regime_short"
        counts = result.group_by(col).len().filter(pl.col(col).is_not_null())
        vals = {
            r["crash_regime_short"]: r["len"] for r in counts.rename({col: "crash_regime_short"}).iter_rows(named=True)
        }
        # "normal" should be the majority (~80%)
        total = sum(vals.values())
        if "normal" in vals:
            assert vals["normal"] / total > 0.5

    def test_crash_regime_custom_quantiles(self):
        # With wider crash band (0.3, 0.7) -> more crash/rally, less normal
        gen_wide = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="crash_regime", kind="discrete")],
            crash_quantiles=(0.3, 0.7),
        )
        gen_narrow = MultiTargetGenerator(
            horizons=[HorizonConfig(name="short", horizon=30)],
            target_types=[TargetType(name="crash_regime", kind="discrete")],
            crash_quantiles=(0.1, 0.9),
        )
        df = _make_ohlcv(500)
        result_wide = gen_wide.generate(df)
        result_narrow = gen_narrow.generate(df)

        col = "target_crash_regime_short"
        normal_wide = result_wide.filter(pl.col(col) == "normal").height
        normal_narrow = result_narrow.filter(pl.col(col) == "normal").height
        # Wider quantiles -> less "normal"
        assert normal_wide < normal_narrow

    def test_fixed_horizon_labeler(self):
        """Test using FixedHorizonLabeler instead of TripleBarrier."""
        gen = MultiTargetGenerator(
            horizons=[
                HorizonConfig(
                    name="short",
                    horizon=30,
                    labeler_cls=FixedHorizonLabeler,
                ),
            ],
            target_types=[TargetType(name="direction", kind="discrete")],
        )
        df = _make_ohlcv(200)
        result = gen.generate(df)
        assert "target_direction_short" in result.columns
