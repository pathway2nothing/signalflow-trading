# IMPORTANT
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.feature.base import FeatureExtractor
from signalflow.core import RawDataView, RawDataType, DataFrameType


@dataclass
class FeatureSet:
    """Polars-first orchestrator for multiple feature extractors.

    Combines independent feature extractors via outer join on (pair, timestamp).
    Each extractor fetches its required data, computes features, and results
    are merged into single DataFrame.

    Key features:
        - Automatic data fetching per extractor
        - Timezone normalization (all → naive)
        - Outer join on (pair, timestamp) for alignment
        - Duplicate feature column detection
        - Consistent index columns across extractors

    Processing flow:
        For each extractor:
            1. Fetch appropriate raw data as Polars
            2. Run extractor.extract()
            3. Normalize timestamps to timezone-naive
            4. Validate index columns present
        Then:
            5. Outer join all results on (pair, timestamp)

    Attributes:
        extractors (list[FeatureExtractor]): Feature extractors to orchestrate.
        parallel (bool): Parallel execution flag (not yet implemented). Default: False.
        pair_col (str): Trading pair column name. Default: "pair".
        ts_col (str): Timestamp column name. Default: "timestamp".

    Example:
        ```python
        from signalflow.feature import FeatureSet, SmaExtractor, RsiExtractor

        # Create feature set
        feature_set = FeatureSet([
            SmaExtractor(window=10, column="close"),
            SmaExtractor(window=20, column="close"),
            RsiExtractor(window=14, column="close")
        ])

        # Extract all features at once
        from signalflow.core import RawDataView
        view = RawDataView(raw=raw_data)
        features = feature_set.extract(view)

        # Result has: pair, timestamp, sma_10, sma_20, rsi_14
        print(features.columns)
        # ['pair', 'timestamp', 'sma_10', 'sma_20', 'rsi_14']
        ```

    Example:
        ```python
        # With multi-timeframe features
        feature_set = FeatureSet([
            # 1-minute features
            SmaExtractor(window=10, offset_window=1),
            # 5-minute features
            SmaExtractor(
                window=10,
                offset_window=5,
                use_resample=True,
                resample_prefix="5m_"
            )
        ])

        features = feature_set.extract(view)
        # Has both 1m and 5m features aligned
        ```

    Note:
        All extractors must use same pair_col and ts_col.
        Feature column names must be unique across extractors.
        Timestamps automatically normalized to timezone-naive.

    See Also:
        FeatureExtractor: Base class for individual extractors.
        RawDataView: Provides data in required format.
    """

    extractors: list[FeatureExtractor]
    parallel: bool = False

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    def __post_init__(self) -> None:
        """Validate extractors configuration.

        Checks:
            - At least one extractor provided
            - All extractors use same pair_col
            - All extractors use same ts_col

        Raises:
            ValueError: If validation fails.
        """
        if not self.extractors:
            raise ValueError("At least one extractor must be provided")

        for ex in self.extractors:
            if getattr(ex, "pair_col", self.pair_col) != self.pair_col:
                raise ValueError(
                    f"All extractors must use pair_col='{self.pair_col}'. "
                    f"{ex.__class__.__name__} uses '{getattr(ex, 'pair_col', None)}'"
                )
            if getattr(ex, "ts_col", self.ts_col) != self.ts_col:
                raise ValueError(
                    f"All extractors must use ts_col='{self.ts_col}'. "
                    f"{ex.__class__.__name__} uses '{getattr(ex, 'ts_col', None)}'"
                )

    def extract(self, raw_data: RawDataView, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Extract and combine features from all extractors.

        Main entry point - orchestrates extraction and merging.

        Processing:
            1. For each extractor:
                - Fetch appropriate data format
                - Run extraction
                - Normalize timestamps
                - Validate output
            2. Outer join all results on (pair, timestamp)
            3. Detect duplicate feature columns

        Args:
            raw_data (RawDataView): View to raw market data.
            context (dict[str, Any] | None): Additional context passed to extractors.

        Returns:
            pl.DataFrame: Combined features with columns:
                - pair, timestamp (index)
                - feature columns from all extractors

        Raises:
            ValueError: If no extractors or duplicate feature columns.
            TypeError: If extractor doesn't return pl.DataFrame.

        Example:
            ```python
            from signalflow.core import RawData, RawDataView

            # Create view
            view = RawDataView(raw=raw_data)

            # Extract features
            features = feature_set.extract(view)

            # Check result
            print(f"Features: {features.columns}")
            print(f"Shape: {features.shape}")

            # With context
            features = feature_set.extract(
                view,
                context={"lookback_bars": 100}
            )
            ```

        Note:
            Outer join means all (pair, timestamp) combinations preserved.
            Missing features filled with null for non-matching timestamps.
        """
        feature_dfs: list[pl.DataFrame] = []

        for extractor in self.extractors:
            input_df = self._get_input_df(raw_data, extractor)

            result_df = extractor.extract(input_df, data_context=context)
            if not isinstance(result_df, pl.DataFrame):
                raise TypeError(
                    f"{extractor.__class__.__name__}.extract must return pl.DataFrame, got {type(result_df)}"
                )

            result_df = self._normalize_index(result_df)

            if self.pair_col not in result_df.columns or self.ts_col not in result_df.columns:
                raise ValueError(
                    f"{extractor.__class__.__name__} returned no index columns "
                    f"('{self.pair_col}', '{self.ts_col}'). "
                    f"FeatureSet requires index columns to combine features."
                )

            feature_dfs.append(result_df)

        return self._combine_features(feature_dfs)

    def _get_input_df(self, raw_data: RawDataView, extractor: FeatureExtractor) -> pl.DataFrame:
        """Fetch input data for extractor in Polars format.

        Determines required data type from extractor.raw_data_type and
        fetches as Polars DataFrame (canonical format).

        Args:
            raw_data (RawDataView): Data view.
            extractor (FeatureExtractor): Extractor needing data.

        Returns:
            pl.DataFrame: Raw data in Polars format.

        Note:
            Always returns Polars (Polars-first design).
            Falls back to string "polars" for backward compatibility.
        """
        raw_data_type = getattr(extractor, "raw_data_type", RawDataType.SPOT)

        try:
            return raw_data.get_data(raw_data_type, DataFrameType.POLARS)
        except TypeError:
            return raw_data.get_data(raw_data_type, "polars")

    def _normalize_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize timestamp to timezone-naive.

        Ensures consistent timezone handling across all extractors.

        Args:
            df (pl.DataFrame): DataFrame to normalize.

        Returns:
            pl.DataFrame: DataFrame with timezone-naive timestamps.
        """
        if self.ts_col in df.columns:
            ts_dtype = df.schema.get(self.ts_col)
            if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
                df = df.with_columns(pl.col(self.ts_col).dt.replace_time_zone(None))
        return df

    def _combine_features(self, feature_dfs: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine feature DataFrames via outer join.

        Merges all feature DataFrames on (pair, timestamp) index.
        Detects and rejects duplicate feature column names.

        Args:
            feature_dfs (list[pl.DataFrame]): Feature DataFrames to combine.

        Returns:
            pl.DataFrame: Combined features with outer join semantics.

        Raises:
            ValueError: If no DataFrames or duplicate feature columns found.

        Example:
            ```python
            # Internal usage
            df1 = pl.DataFrame({"pair": ["BTC"], "timestamp": [t1], "sma_10": [45000]})
            df2 = pl.DataFrame({"pair": ["BTC"], "timestamp": [t1], "rsi_14": [50]})
            combined = self._combine_features([df1, df2])
            # Result: pair, timestamp, sma_10, rsi_14
            ```

        Note:
            Outer join preserves all (pair, timestamp) from all extractors.
            Duplicate columns trigger error - use unique prefixes.
        """
        if not feature_dfs:
            raise ValueError("No feature DataFrames to combine")

        combined = feature_dfs[0]

        for right in feature_dfs[1:]:
            right_feature_cols = [c for c in right.columns if c not in (self.pair_col, self.ts_col)]
            dup = set(right_feature_cols).intersection(set(combined.columns))
            if dup:
                raise ValueError(
                    f"Duplicate feature columns during FeatureSet combine: {sorted(dup)}. "
                    f"Rename features or set unique prefixes."
                )

            combined = combined.join(right, on=[self.pair_col, self.ts_col], how="outer", coalesce=True)

        return combined