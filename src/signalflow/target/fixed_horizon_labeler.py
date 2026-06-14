from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.enums import Signal as SignalType
from signalflow.target._soft_helpers import signed_tercile_soft
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


@dataclass
@register_target("fixed_horizon_labeler")
class FixedHorizonLabeler(Labeler):
    """
    Fixed-Horizon Labeling:
      label[t0] = sign(close[t0 + horizon] - close[t0])

    If signals provided, labels are written only on signal rows,
    while horizon is computed on full series (per pair).
    """

    soft_classes: ClassVar[tuple[str, ...]] = (
        SignalType.FALL.value,
        SignalType.NONE.value,
        SignalType.RISE.value,
    )

    price_col: str = "close"
    horizon: int = 60

    soft_noise_floor: float = 0.0
    """Indifference band for soft labels (log-return units).

    With the default ``0.0`` the soft cut sits exactly at zero and probabilities
    collapse to the hard ``sign(fwd_ret)`` decision as ``softness_k`` grows.
    Set to e.g. ``0.001`` (≈10 bps) to put a no-change bucket around zero so
    bars with tiny forward moves get appreciable ``p_none`` mass.
    """

    meta_columns: tuple[str, ...] = ("t1", "ret")

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")

        if group_df.height == 0:
            return group_df

        h = int(self.horizon)
        price = pl.col(self.price_col)
        future_price = price.shift(-h)

        df = group_df.with_columns(future_price.alias("_future_price"))

        label_expr = (
            pl.when(
                pl.col("_future_price").is_null()
                | pl.col(self.price_col).is_null()
                | (pl.col(self.price_col) <= 0)
                | (pl.col("_future_price") <= 0)
            )
            .then(pl.lit(SignalType.NONE.value))
            .when(pl.col("_future_price") > pl.col(self.price_col))
            .then(pl.lit(SignalType.RISE.value))
            .when(pl.col("_future_price") < pl.col(self.price_col))
            .then(pl.lit(SignalType.FALL.value))
            .otherwise(pl.lit(SignalType.NONE.value))
        )

        df = df.with_columns(label_expr.alias(self.out_col))

        if self.include_meta:
            df = df.with_columns(
                [
                    pl.col(self.ts_col).shift(-h).alias("t1"),
                    pl.when(
                        pl.col("_future_price").is_not_null()
                        & (pl.col(self.price_col) > 0)
                        & (pl.col("_future_price") > 0)
                    )
                    .then((pl.col("_future_price") / pl.col(self.price_col)).log())
                    .otherwise(pl.lit(None))
                    .alias("ret"),
                ]
            )

        df = df.drop("_future_price")

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)

        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft labels: sigmoid over the signed forward log-return."""
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if group_df.height == 0:
            return group_df

        h = int(self.horizon)
        price = pl.col(self.price_col)
        future_price = price.shift(-h)

        df = group_df.with_columns(future_price.alias("_future_price"))
        df = df.with_columns(
            pl.when(
                pl.col("_future_price").is_null()
                | pl.col(self.price_col).is_null()
                | (pl.col(self.price_col) <= 0)
                | (pl.col("_future_price") <= 0)
            )
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("_future_price") / pl.col(self.price_col)).log())
            .alias("_fwd_logret")
        )

        p_fall, p_none, p_rise = signed_tercile_soft(
            pl.col("_fwd_logret"),
            neg_threshold=self.soft_noise_floor,
            pos_threshold=self.soft_noise_floor,
            k=self.softness_k,
        )
        df = df.with_columns(
            p_fall.alias(f"{self.soft_col_prefix}{SignalType.FALL.value}"),
            p_none.alias(f"{self.soft_col_prefix}{SignalType.NONE.value}"),
            p_rise.alias(f"{self.soft_col_prefix}{SignalType.RISE.value}"),
        )
        df = df.drop(["_future_price", "_fwd_logret"])
        return df
