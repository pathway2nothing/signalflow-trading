from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from signalflow.core import RawDataType, RollingAggregator, SfComponentType
from typing import ClassVar


@dataclass
class FeatureExtractor(ABC):
    """Base class for Polars-first feature extraction.

    Extracts technical indicators and derived features from raw OHLCV data
    with optional sliding window resampling for multi-timeframe features.

    Key features:
        - Polars-native for performance
        - Optional sliding window resampling (e.g., 5m bars from 1m bars)
        - Per-pair, per-offset-window processing
        - Length-preserving operations
        - Automatic projection (keep only new features)

    Processing pipeline:
        1. Sort by (pair, timestamp)
        2. Add resample_offset column
        3. (optional) Apply sliding resample
        4. (optional) Filter to last offset
        5. Group by (pair, resample_offset) and compute features
        6. Sort output
        7. Project columns (keep input or features only)

    Attributes:
        offset_window (int): Sliding window size in bars. Default: 1.
        compute_last_offset (bool): Keep only last offset. Default: False.
        pair_col (str): Trading pair column. Default: "pair".
        ts_col (str): Timestamp column. Default: "timestamp".
        offset_col (str): Offset tracking column. Default: "resample_offset".
        use_resample (bool): Apply sliding resample. Default: False.
        resample_mode (Literal["add", "replace"]): Resample mode. Default: "add".
        resample_prefix (str | None): Prefix for resampled columns. Default: None.
        raw_data_type (RawDataType): Type of raw data. Default: SPOT.
        component_type (ClassVar[SfComponentType]): Always FEATURE_EXTRACTOR.
        keep_input_columns (bool): Keep all input columns. Default: False.

    Example:
        ```python
        from signalflow.feature import FeatureExtractor
        import polars as pl

        class RsiExtractor(FeatureExtractor):
            '''RSI indicator extractor'''
            
            def __init__(self, window: int = 14, column: str = "close"):
                super().__init__()
                self.window = window
                self.column = column
            
            def compute_group(self, group_df, data_context=None):
                # Compute RSI per group
                delta = group_df.select(pl.col(self.column).diff().alias("delta"))
                
                gain = delta.select(
                    pl.when(pl.col("delta") > 0)
                    .then(pl.col("delta"))
                    .otherwise(0)
                    .alias("gain")
                )
                
                loss = delta.select(
                    pl.when(pl.col("delta") < 0)
                    .then(-pl.col("delta"))
                    .otherwise(0)
                    .alias("loss")
                )
                
                avg_gain = gain.select(
                    pl.col("gain").rolling_mean(self.window).alias("avg_gain")
                )
                
                avg_loss = loss.select(
                    pl.col("loss").rolling_mean(self.window).alias("avg_loss")
                )
                
                rs = avg_gain.select(
                    (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs")
                )
                
                rsi = group_df.with_columns([
                    (100 - (100 / (1 + rs.get_column("rs")))).alias(f"rsi_{self.window}")
                ])
                
                return rsi

        # Usage
        extractor = RsiExtractor(window=14)
        features = extractor.extract(ohlcv_df)
        ```

    Note:
        compute_group() must preserve row count (length-preserving).
        All timestamps must be timezone-naive.
        For multi-timeframe features, use use_resample=True.

    See Also:
        RollingAggregator: Sliding window resampler.
        FeatureSet: Orchestrates multiple extractors.
    """

    offset_window: int = 1
    compute_last_offset: bool = False

    pair_col: str = "pair"
    ts_col: str = "timestamp"
    offset_col: str = "resample_offset"

    use_resample: bool = False
    resample_mode: Literal["add", "replace"] = "add"
    resample_prefix: str | None = None
    raw_data_type: RawDataType = RawDataType.SPOT
    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE_EXTRACTOR
    keep_input_columns: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If offset_window <= 0, invalid resample_mode, or wrong offset_col.
            TypeError: If column names not strings.
        """
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")

        if self.resample_mode not in ("add", "replace"):
            raise ValueError(f"Invalid resample_mode: {self.resample_mode}")

        if self.offset_col != RollingAggregator.OFFSET_COL:
            raise ValueError(
                f"offset_col must be '{RollingAggregator.OFFSET_COL}', got '{self.offset_col}'"
            )

        if not isinstance(self.pair_col, str) or not isinstance(self.ts_col, str) or not isinstance(self.offset_col, str):
            raise TypeError("pair_col/ts_col/offset_col must be str")

    @property
    def _resampler(self) -> RollingAggregator:
        """Get configured RollingAggregator instance.

        Returns:
            RollingAggregator: Resampler with current configuration.
        """
        return RollingAggregator(
            offset_window=self.offset_window,
            ts_col=self.ts_col,
            pair_col=self.pair_col,
            mode=self.resample_mode,
            prefix=self.resample_prefix,
            raw_data_type=self.raw_data_type,
        )

    def extract(self, df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Extract features from input DataFrame.

        Main entry point - handles sorting, resampling, grouping, and projection.

        Processing pipeline:
            1. Validate input (required columns)
            2. Sort by (pair, timestamp)
            3. Add resample_offset column if missing
            4. (optional) Apply sliding resample
            5. (optional) Filter to last offset
            6. Group by (pair, resample_offset) and compute features
            7. Sort output
            8. Project to output columns

        Args:
            df (pl.DataFrame): Input OHLCV data with pair and timestamp columns.
            data_context (dict[str, Any] | None): Additional context for computation.

        Returns:
            pl.DataFrame: Features DataFrame with columns:
                - pair, timestamp (always included)
                - feature columns (from compute_group)

        Raises:
            TypeError: If df not pl.DataFrame or compute_group returns wrong type.
            ValueError: If compute_group changes row count or columns missing.

        Example:
            ```python
            # Basic extraction
            features = extractor.extract(ohlcv_df)

            # With resampling (5m from 1m)
            extractor = RsiExtractor(
                window=14,
                offset_window=5,
                use_resample=True
            )
            features = extractor.extract(ohlcv_df)

            # Keep input columns
            extractor.keep_input_columns = True
            features_with_ohlcv = extractor.extract(ohlcv_df)
            ```

        Note:
            Only accepts pl.DataFrame (Polars-first design).
            Use PandasFeatureExtractor adapter for Pandas data.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} is polars-first and accepts only pl.DataFrame. "
                f"Got: {type(df)}. Use an adapter for other dataframe types."
            )
        self._validate_input(df)

        df0 = df.sort([self.pair_col, self.ts_col])

        if self.offset_col not in df0.columns:
            df0 = self._resampler.add_offset_column(df0)

        if self.use_resample:
            df0 = self._resampler.resample(df0)

        if self.compute_last_offset:
            last_off = self._resampler.get_last_offset(df0)
            df0 = df0.filter(pl.col(self.offset_col) == last_off)

        prepared_cols = set(df0.columns)
        inferred_features: set[str] = set()

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            nonlocal inferred_features

            in_cols = set(g.columns)
            out = self.compute_group(g, data_context=data_context)

            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_pl_group must return pl.DataFrame")

            if out.height != g.height:
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
                )

            if not inferred_features:
                inferred_features = set(out.columns) - in_cols

            return out

        out = (
            df0.group_by(self.pair_col, self.offset_col, maintain_order=True)
            .map_groups(_wrapped)
            .sort([self.pair_col, self.ts_col])
        )

        if self.keep_input_columns:
            return out

        feature_cols = sorted(set(out.columns) - prepared_cols)
        keep_cols = [self.pair_col, self.ts_col] + feature_cols

        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        """Compute features for single (pair, resample_offset) group.

        Core feature extraction logic - must be implemented by subclasses.

        CRITICAL: Must preserve row count (len(output) == len(input)).
        Should preserve ordering within group.

        Args:
            group_df (pl.DataFrame): Single group's data, sorted by timestamp.
            data_context (dict[str, Any] | None): Additional context.

        Returns:
            pl.DataFrame: Same length as input with added feature columns.

        Example:
            ```python
            def compute_group(self, group_df, data_context=None):
                # Simple moving average
                return group_df.with_columns([
                    pl.col("close")
                    .rolling_mean(self.window)
                    .alias(f"sma_{self.window}")
                ])
            
            # Multiple features
            def compute_group(self, group_df, data_context=None):
                return group_df.with_columns([
                    pl.col("close").rolling_mean(10).alias("sma_10"),
                    pl.col("close").rolling_mean(20).alias("sma_20"),
                    pl.col("high").rolling_max(14).alias("high_14"),
                    pl.col("low").rolling_min(14).alias("low_14")
                ])
            ```

        Note:
            Output must have same height as input (length-preserving).
            Use rolling operations for windowed features.
            First N-1 bars may have null values for N-period indicators.
        """
        raise NotImplementedError

    def _validate_input(self, df: pl.DataFrame) -> None:
        """Validate input DataFrame has required columns.

        Args:
            df (pl.DataFrame): Input to validate.

        Raises:
            ValueError: If required columns missing.
        """
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")