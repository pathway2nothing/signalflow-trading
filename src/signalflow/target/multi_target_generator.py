"""Multi-horizon, multi-target label generation for informativeness analysis.

Wraps existing Labeler subclasses to produce multiple label columns
at different horizons and target types simultaneously.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from signalflow.core import SignalType
from signalflow.target.base import Labeler
from signalflow.target.triple_barrier_labeler import TripleBarrierLabeler


@dataclass
class HorizonConfig:
    """Configuration for a single prediction horizon.

    Attributes:
        name: Human-readable name (e.g., "short", "mid", "long").
        horizon: Number of bars for the horizon.
        labeler_cls: Labeler class to use for direction targets.
        labeler_kwargs: Extra kwargs passed to the labeler constructor.
    """

    name: str
    horizon: int
    labeler_cls: type[Labeler] = TripleBarrierLabeler
    labeler_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetType:
    """Defines a target type derived from OHLCV data.

    Attributes:
        name: Target name (e.g., "direction", "return_magnitude", "volume_regime").
        kind: "discrete" or "continuous" — determines MI computation method.
    """

    name: str
    kind: str  # "discrete" | "continuous"


DEFAULT_HORIZONS: list[HorizonConfig] = [
    HorizonConfig(name="short", horizon=60),
    HorizonConfig(name="mid", horizon=360),
    HorizonConfig(name="long", horizon=1440),
]

DEFAULT_TARGET_TYPES: list[TargetType] = [
    TargetType(name="direction", kind="discrete"),
    TargetType(name="return_magnitude", kind="continuous"),
    TargetType(name="volume_regime", kind="discrete"),
    TargetType(name="crash_regime", kind="discrete"),
]


@dataclass
class MultiTargetGenerator:
    """Generates multiple targets at multiple horizons from OHLCV data.

    For each (horizon, target_type) combination, adds a column to the
    DataFrame. Column naming convention: ``target_{target_name}_{horizon_name}``.

    Direction targets use the existing Labeler infrastructure.
    Return magnitude is ``|log(close[t+h] / close[t])|``.
    Volume regime discretizes ``volume / sma(volume)`` into HIGH/MED/LOW.
    Crash regime classifies forward return into crash/rally/normal.

    Attributes:
        horizons: List of HorizonConfig.
        target_types: List of TargetType to generate.
        volume_window: Rolling window for volume SMA baseline.
        volume_quantiles: (low, high) thresholds for volume regime.
        crash_quantiles: (crash, rally) quantile thresholds for crash regime.
        pair_col: Trading pair column name.
        ts_col: Timestamp column name.
        price_col: Price column name.
    """

    horizons: list[HorizonConfig] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    target_types: list[TargetType] = field(default_factory=lambda: list(DEFAULT_TARGET_TYPES))

    volume_window: int = 60
    volume_quantiles: tuple[float, float] = (0.33, 0.67)
    crash_quantiles: tuple[float, float] = (0.1, 0.9)

    pair_col: str = "pair"
    ts_col: str = "timestamp"
    price_col: str = "close"

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate all target columns.

        Args:
            df: OHLCV DataFrame with pair, timestamp, open, high, low, close, volume.

        Returns:
            Original DataFrame with added ``target_*`` columns.
        """
        self._validate(df)

        target_type_map = {t.name: t for t in self.target_types}

        if "direction" in target_type_map:
            df = self._generate_direction(df)

        if "return_magnitude" in target_type_map:
            df = self._generate_return_magnitude(df)

        if "volume_regime" in target_type_map:
            df = self._generate_volume_regime(df)

        if "crash_regime" in target_type_map:
            df = self._generate_crash_regime(df)

        return df

    def target_columns(self) -> list[dict[str, str]]:
        """Return metadata for all generated target columns.

        Returns:
            List of dicts with keys: column, horizon, target_type, kind.
        """
        result = []
        for h in self.horizons:
            for t in self.target_types:
                result.append(
                    {
                        "column": f"target_{t.name}_{h.name}",
                        "horizon": h.name,
                        "target_type": t.name,
                        "kind": t.kind,
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Direction targets via existing Labelers
    # ------------------------------------------------------------------

    def _generate_direction(self, df: pl.DataFrame) -> pl.DataFrame:
        for h in self.horizons:
            col_name = f"target_direction_{h.name}"
            labeler = self._create_labeler(h)
            labeled = labeler.compute(df)

            label_series = labeled.get_column(labeler.out_col).alias(col_name)
            df = df.with_columns(label_series)

            logger.debug(f"Generated direction target: {col_name}")

        return df

    def _create_labeler(self, h: HorizonConfig) -> Labeler:
        """Instantiate a labeler for the given horizon."""
        kwargs = {
            "price_col": self.price_col,
            "pair_col": self.pair_col,
            "ts_col": self.ts_col,
            "keep_input_columns": True,
            "mask_to_signals": False,
            "include_meta": False,
        }

        kwargs["horizon"] = h.horizon

        kwargs.update(h.labeler_kwargs)
        return h.labeler_cls(**kwargs)

    # ------------------------------------------------------------------
    # Return magnitude: |log(close[t+h] / close[t])|
    # ------------------------------------------------------------------

    def _generate_return_magnitude(self, df: pl.DataFrame) -> pl.DataFrame:
        for h in self.horizons:
            col_name = f"target_return_magnitude_{h.name}"

            df = df.with_columns(
                df.group_by(self.pair_col, maintain_order=True)
                .agg((pl.col(self.price_col).shift(-h.horizon) / pl.col(self.price_col)).log().abs().alias(col_name))
                .sort(self.pair_col)
                .get_column(col_name)
                .explode()
                .alias(col_name)
            )

            logger.debug(f"Generated return magnitude target: {col_name}")

        return df

    # ------------------------------------------------------------------
    # Volume regime: HIGH / MED / LOW
    # ------------------------------------------------------------------

    def _generate_volume_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        for h in self.horizons:
            col_name = f"target_volume_regime_{h.name}"

            df = df.with_columns(
                df.group_by(self.pair_col, maintain_order=True)
                .agg(self._volume_regime_expr(h.horizon).alias(col_name))
                .sort(self.pair_col)
                .get_column(col_name)
                .explode()
                .alias(col_name)
            )

            logger.debug(f"Generated volume regime target: {col_name}")

        return df

    def _volume_regime_expr(self, horizon: int) -> pl.Expr:
        """Polars expression for volume regime classification.

        Computes forward-looking average volume ratio vs current rolling SMA,
        then discretizes into HIGH/MED/LOW based on quantile thresholds.
        """
        vol = pl.col("volume")
        vol_sma = vol.rolling_mean(window_size=self.volume_window)
        vol_ratio = vol / vol_sma

        # Use forward-looking mean of volume_ratio over the horizon
        forward_vol_ratio = vol_ratio.shift(-horizon)

        low_q, high_q = self.volume_quantiles
        return (
            pl.when(forward_vol_ratio.is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(forward_vol_ratio >= forward_vol_ratio.quantile(high_q))
            .then(pl.lit("HIGH"))
            .when(forward_vol_ratio <= forward_vol_ratio.quantile(low_q))
            .then(pl.lit("LOW"))
            .otherwise(pl.lit("MED"))
        )

    # ------------------------------------------------------------------
    # Crash regime: crash / normal / rally
    # ------------------------------------------------------------------

    def _generate_crash_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        for h in self.horizons:
            col_name = f"target_crash_regime_{h.name}"

            df = df.with_columns(
                df.group_by(self.pair_col, maintain_order=True)
                .agg(self._crash_regime_expr(h.horizon).alias(col_name))
                .sort(self.pair_col)
                .get_column(col_name)
                .explode()
                .alias(col_name)
            )

            logger.debug(f"Generated crash regime target: {col_name}")

        return df

    def _crash_regime_expr(self, horizon: int) -> pl.Expr:
        """Polars expression for crash/rally regime classification.

        Computes forward log-return over ``horizon`` bars, then discretizes
        into crash/rally/normal based on quantile thresholds.
        """
        price = pl.col(self.price_col)
        forward_return = (price.shift(-horizon) / price).log()

        crash_q, rally_q = self.crash_quantiles
        return (
            pl.when(forward_return.is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(forward_return <= forward_return.quantile(crash_q))
            .then(pl.lit("crash"))
            .when(forward_return >= forward_return.quantile(rally_q))
            .then(pl.lit("rally"))
            .otherwise(pl.lit("normal"))
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, df: pl.DataFrame) -> None:
        required = {self.pair_col, self.ts_col, self.price_col}
        target_type_names = {t.name for t in self.target_types}
        if "volume_regime" in target_type_names:
            required.add("volume")

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if not self.horizons:
            raise ValueError("At least one horizon is required")
        if not self.target_types:
            raise ValueError("At least one target type is required")
