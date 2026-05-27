"""Flash move labeler.

Labels bars where the forward absolute return over a *short* horizon
exceeds a multiple of trailing volatility — captures slippage/microstructure
events (flash crashes, flash pumps) rather than slow regime changes.

Complements :class:`AnomalyLabeler`, which uses longer horizons and looks
for extreme regime moves. This labeler is tuned for sub-10-bar bursts that
matter for execution risk.

Implementation uses pure Polars expressions for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import labeler
from signalflow.core.enums import SignalCategory
from signalflow.target._soft_helpers import signed_tercile_soft
from signalflow.target.base import Labeler


@dataclass
@labeler("flash_move")
class FlashMoveLabeler(Labeler):
    """Label bars by extreme forward short-horizon returns.

    Algorithm:
        1. log_returns = ln(close[t] / close[t-1])
        2. past_vol    = rolling_std(log_returns, vol_window)
        3. fwd_logret  = ln(close[t+flash_horizon] / close[t])
        4. threshold   = sigma_multiplier * past_vol * sqrt(flash_horizon)
        5. If  fwd_logret >  threshold -> ``"flash_up"``
           If  fwd_logret < -threshold -> ``"flash_dn"``
           Otherwise                    -> ``"normal"``

    Attributes:
        price_col: Price column. Default: ``"close"``.
        flash_horizon: Short forward window. Default: ``10`` bars.
        vol_window: Trailing window for vol baseline. Default: ``1440``.
        sigma_multiplier: Magnitude in standard deviations to qualify as
            flash. Default: ``3.0``.

    Example:
        ```python
        from signalflow.target.flash_move_labeler import FlashMoveLabeler

        labeler = FlashMoveLabeler(
            flash_horizon=10,
            vol_window=1440,
            sigma_multiplier=3.0,
            mask_to_signals=False,
        )
        result = labeler.compute(ohlcv_df)
        ```
    """

    signal_category: SignalCategory = SignalCategory.ANOMALY

    soft_classes: ClassVar[tuple[str, ...]] = ("flash_dn", "normal", "flash_up")

    price_col: str = "close"
    flash_horizon: int = 10
    vol_window: int = 1440
    sigma_multiplier: float = 3.0

    meta_columns: tuple[str, ...] = ("flash_ret", "flash_threshold")

    def __post_init__(self) -> None:
        if self.flash_horizon <= 0:
            raise ValueError("flash_horizon must be > 0")
        if self.vol_window <= 1:
            raise ValueError("vol_window must be > 1")
        if self.sigma_multiplier <= 0:
            raise ValueError("sigma_multiplier must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        import math

        price = pl.col(self.price_col)
        log_ret = (price / price.shift(1)).log()

        df = group_df.with_columns(log_ret.alias("_log_ret"))

        df = df.with_columns(
            pl.col("_log_ret").rolling_std(self.vol_window, min_samples=2).alias("_past_vol"),
            (price.shift(-self.flash_horizon) / price).log().alias("_fwd_logret"),
        )

        df = df.with_columns(
            (pl.col("_past_vol") * self.sigma_multiplier * math.sqrt(self.flash_horizon)).alias("_threshold"),
        )

        label_expr = (
            pl.when(pl.col("_fwd_logret").is_null() | pl.col("_threshold").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("_fwd_logret") > pl.col("_threshold"))
            .then(pl.lit("flash_up"))
            .when(pl.col("_fwd_logret") < -pl.col("_threshold"))
            .then(pl.lit("flash_dn"))
            .otherwise(pl.lit("normal"))
            .alias(self.out_col)
        )

        df = df.with_columns(label_expr)

        if self.include_meta:
            df = df.with_columns(
                pl.col("_fwd_logret").alias("flash_ret"),
                pl.col("_threshold").alias("flash_threshold"),
            )

        df = df.drop(["_log_ret", "_past_vol", "_fwd_logret", "_threshold"])

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft triple ``(p_flash_dn, p_normal, p_flash_up)`` over short-horizon z.

        Reduces the same ``fwd_logret`` vs ``±sigma_multiplier · past_vol · √h``
        cut to a sigmoid in z-units (``fwd_logret / (past_vol · √h)``), so the
        decision boundary becomes ``±sigma_multiplier``.
        """
        if group_df.height == 0:
            return group_df
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        import math

        price = pl.col(self.price_col)
        log_ret = (price / price.shift(1)).log()
        df = group_df.with_columns(log_ret.alias("_log_ret"))
        df = df.with_columns(
            pl.col("_log_ret").rolling_std(self.vol_window, min_samples=2).alias("_past_vol"),
            (price.shift(-self.flash_horizon) / price).log().alias("_fwd_logret"),
        )
        scale = math.sqrt(self.flash_horizon)
        z = (
            pl.when(pl.col("_past_vol") > 0)
            .then(pl.col("_fwd_logret") / (pl.col("_past_vol") * scale))
            .otherwise(pl.lit(None))
        )
        df = df.with_columns(z.alias("_z"))

        p_dn, p_mid, p_up = signed_tercile_soft(
            pl.col("_z"),
            neg_threshold=self.sigma_multiplier,
            pos_threshold=self.sigma_multiplier,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_dn.alias(f"{self.soft_col_prefix}flash_dn"),
            p_mid.alias(f"{self.soft_col_prefix}normal"),
            p_up.alias(f"{self.soft_col_prefix}flash_up"),
        )
        df = df.drop(["_log_ret", "_past_vol", "_fwd_logret", "_z"])
        return df
