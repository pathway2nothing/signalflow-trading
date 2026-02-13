"""Aggregated Open Interest feature.

Computes market-wide aggregated open interest across all pairs,
useful for detecting market-wide sentiment shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from signalflow.core import SfComponentType, sf_component
from signalflow.feature.base import GlobalFeature

if TYPE_CHECKING:
    from signalflow.core import RawData


@dataclass
@sf_component(name="aggregated_oi")
class AggregatedOpenInterest(GlobalFeature):
    """Computes aggregated open interest across all pairs.

    This is a GlobalFeature that sums open interest across all trading pairs
    at each timestamp, providing a market-wide positioning indicator.

    Outputs:
        - agg_oi: Total open interest across all pairs
        - agg_oi_change: Percentage change in aggregated OI
        - agg_oi_zscore: Z-score of aggregated OI (normalized)

    Attributes:
        oi_col: Column name for open interest data.
        zscore_window: Rolling window for z-score calculation.
        include_pair_count: If True, also output n_pairs column.

    Example:
        ```python
        from signalflow.feature import AggregatedOpenInterest

        feature = AggregatedOpenInterest(zscore_window=21)
        df_with_agg_oi = feature.compute(raw_df)

        # Result has agg_oi, agg_oi_change, agg_oi_zscore joined to each row
        ```
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE
    requires: ClassVar[list[str]] = ["{oi_col}"]
    outputs: ClassVar[list[str]] = ["agg_oi", "agg_oi_change", "agg_oi_zscore"]

    oi_col: str = "open_interest"
    zscore_window: int = 21  # ~7 days at 8h timeframe
    include_pair_count: bool = False

    def compute(
        self,
        df: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute aggregated open interest and join back to input.

        Args:
            df: Input DataFrame with pair, timestamp, and open_interest columns.
            context: Optional context dict (unused).

        Returns:
            Input DataFrame with agg_oi, agg_oi_change, agg_oi_zscore columns added.
        """
        # Step 1: Aggregate OI by timestamp
        agg_oi = (
            df.group_by(self.ts_col)
            .agg([
                pl.col(self.oi_col).sum().alias("agg_oi"),
                pl.col(self.group_col).n_unique().alias("n_pairs"),
            ])
            .sort(self.ts_col)
        )

        # Step 2: Compute derived features
        agg_oi = agg_oi.with_columns([
            # Percentage change
            (pl.col("agg_oi") / pl.col("agg_oi").shift(1) - 1).alias("agg_oi_change"),
            # Rolling stats for z-score
            pl.col("agg_oi")
            .rolling_mean(window_size=self.zscore_window)
            .alias("_mean"),
            pl.col("agg_oi")
            .rolling_std(window_size=self.zscore_window)
            .alias("_std"),
        ])

        # Z-score with safe division
        agg_oi = agg_oi.with_columns(
            pl.when(pl.col("_std") > 0)
            .then((pl.col("agg_oi") - pl.col("_mean")) / pl.col("_std"))
            .otherwise(pl.lit(0.0))
            .alias("agg_oi_zscore")
        ).drop(["_mean", "_std"])

        # Step 3: Select columns to join
        join_cols = [self.ts_col, "agg_oi", "agg_oi_change", "agg_oi_zscore"]
        if self.include_pair_count:
            join_cols.append("n_pairs")
        else:
            agg_oi = agg_oi.drop("n_pairs")

        # Step 4: Join back to original DataFrame
        result = df.join(
            agg_oi.select(join_cols),
            on=self.ts_col,
            how="left",
        )

        return result

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable z-score output."""
        return self.zscore_window


@dataclass
@sf_component(name="aggregated_oi_multi_source")
class AggregatedOpenInterestMultiSource(GlobalFeature):
    """Aggregated open interest from multiple data sources (exchanges).

    Combines open interest from multiple exchanges into a single
    market-wide aggregated metric.

    Supports two usage patterns:
    1. compute(df) - DataFrame with source column (legacy, merged data)
    2. compute_from_raw(raw) - RawData with nested structure (recommended)

    Attributes:
        source_col: Column name identifying the data source/exchange.
        oi_col: Column name for open interest data.
        zscore_window: Rolling window for z-score calculation.
        sources: List of source names to include. If None, use all.
        data_type: Data type key in RawData (for compute_from_raw).

    Example:
        ```python
        from signalflow.feature import AggregatedOpenInterestMultiSource
        from signalflow.data import RawDataFactory

        # Load multi-source data
        raw = RawDataFactory.from_stores(
            stores={
                "binance": binance_store,
                "okx": okx_store,
            },
            pairs=["BTCUSDT"],
            start=start,
            end=end,
        )

        # Use with new RawData API (recommended)
        feature = AggregatedOpenInterestMultiSource(
            sources=["binance", "okx"],
            data_type="perpetual",
        )
        result = feature.compute_from_raw(raw)

        # Legacy: merged DataFrame with source column
        feature = AggregatedOpenInterestMultiSource(sources=["binance", "okx"])
        result = feature.compute(combined_df)
        ```
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE
    requires: ClassVar[list[str]] = ["{oi_col}", "{source_col}"]
    outputs: ClassVar[list[str]] = [
        "agg_oi_total",
        "agg_oi_change",
        "agg_oi_zscore",
    ]

    source_col: str = "source"
    oi_col: str = "open_interest"
    zscore_window: int = 21
    include_per_source: bool = False  # Include per-source breakdown
    data_type: str = "perpetual"  # For compute_from_raw

    def compute_from_raw(
        self,
        raw: "RawData",
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute aggregated OI directly from RawData.

        Uses hierarchical RawData access (raw.perpetual.binance) to
        iterate over sources without merging into a single DataFrame.

        Args:
            raw: RawData container with nested structure.
            context: Optional context dict (unused).

        Returns:
            DataFrame with timestamp, agg_oi_total, agg_oi_change, agg_oi_zscore.
        """
        # Collect per-timestamp aggregates from each source
        source_aggs: list[pl.DataFrame] = []

        for source_name, df in self.iter_sources(raw, self.data_type):
            if self.oi_col not in df.columns:
                continue

            # Aggregate OI per timestamp for this source
            agg = (
                df.group_by(self.ts_col)
                .agg([
                    pl.col(self.oi_col).sum().alias("_oi"),
                ])
                .with_columns(pl.lit(source_name).alias("_source"))
            )
            source_aggs.append(agg)

        if not source_aggs:
            return pl.DataFrame(schema={
                self.ts_col: pl.Datetime("us"),
                "agg_oi_total": pl.Float64,
                "agg_oi_change": pl.Float64,
                "agg_oi_zscore": pl.Float64,
            })

        # Combine and aggregate across sources
        combined = pl.concat(source_aggs)
        agg_oi = (
            combined.group_by(self.ts_col)
            .agg([
                pl.col("_oi").sum().alias("agg_oi_total"),
                pl.col("_source").n_unique().alias("n_sources"),
            ])
            .sort(self.ts_col)
        )

        # Per-source breakdown if requested
        if self.include_per_source:
            per_source = combined.pivot(
                on="_source",
                index=self.ts_col,
                values="_oi",
            )
            source_cols = [c for c in per_source.columns if c != self.ts_col]
            rename_map = {c: f"agg_oi_{c}" for c in source_cols}
            per_source = per_source.rename(rename_map)
            agg_oi = agg_oi.join(per_source, on=self.ts_col, how="left")

        # Derived features
        agg_oi = agg_oi.with_columns([
            (pl.col("agg_oi_total") / pl.col("agg_oi_total").shift(1) - 1).alias(
                "agg_oi_change"
            ),
            pl.col("agg_oi_total")
            .rolling_mean(window_size=self.zscore_window)
            .alias("_mean"),
            pl.col("agg_oi_total")
            .rolling_std(window_size=self.zscore_window)
            .alias("_std"),
        ])

        agg_oi = agg_oi.with_columns(
            pl.when(pl.col("_std") > 0)
            .then((pl.col("agg_oi_total") - pl.col("_mean")) / pl.col("_std"))
            .otherwise(pl.lit(0.0))
            .alias("agg_oi_zscore")
        ).drop(["_mean", "_std", "n_sources"])

        return agg_oi

    def compute(
        self,
        df: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute aggregated OI from merged DataFrame (legacy).

        Args:
            df: Input DataFrame with source, pair, timestamp, open_interest.
            context: Optional context dict (unused).

        Returns:
            DataFrame with aggregated OI columns added.
        """
        # Filter to specified sources if provided
        if self.sources is not None:
            df_filtered = df.filter(pl.col(self.source_col).is_in(self.sources))
        else:
            df_filtered = df

        # Step 1: Aggregate OI by timestamp (across all sources and pairs)
        agg_oi = (
            df_filtered.group_by(self.ts_col)
            .agg([
                pl.col(self.oi_col).sum().alias("agg_oi_total"),
                pl.col(self.source_col).n_unique().alias("n_sources"),
                pl.col(self.group_col).n_unique().alias("n_pairs"),
            ])
            .sort(self.ts_col)
        )

        # Per-source aggregation if requested
        if self.include_per_source:
            per_source = (
                df_filtered.group_by([self.ts_col, self.source_col])
                .agg(pl.col(self.oi_col).sum().alias("source_oi"))
                .pivot(
                    on=self.source_col,
                    index=self.ts_col,
                    values="source_oi",
                )
            )
            # Rename pivoted columns
            source_cols = [c for c in per_source.columns if c != self.ts_col]
            rename_map = {c: f"agg_oi_{c}" for c in source_cols}
            per_source = per_source.rename(rename_map)
            agg_oi = agg_oi.join(per_source, on=self.ts_col, how="left")

        # Step 2: Derived features
        agg_oi = agg_oi.with_columns([
            (pl.col("agg_oi_total") / pl.col("agg_oi_total").shift(1) - 1).alias(
                "agg_oi_change"
            ),
            pl.col("agg_oi_total")
            .rolling_mean(window_size=self.zscore_window)
            .alias("_mean"),
            pl.col("agg_oi_total")
            .rolling_std(window_size=self.zscore_window)
            .alias("_std"),
        ])

        agg_oi = agg_oi.with_columns(
            pl.when(pl.col("_std") > 0)
            .then((pl.col("agg_oi_total") - pl.col("_mean")) / pl.col("_std"))
            .otherwise(pl.lit(0.0))
            .alias("agg_oi_zscore")
        ).drop(["_mean", "_std", "n_sources", "n_pairs"])

        # Step 3: Join back to original
        result = df.join(agg_oi, on=self.ts_col, how="left")

        return result

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable z-score output."""
        return self.zscore_window
