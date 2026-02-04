from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl
from signalflow.core.enums import RawDataType


@dataclass
class RollingAggregator:
    """Offset (sliding) resampler for raw market data.

    Computes rolling aggregates over a sliding window of bars for each trading pair.
    For each row at time t, aggregates over the last `offset_window` rows: [t-(k-1), ..., t].

    Key features:
        - Length-preserving: len(output) == len(input)
        - Per-pair processing: Each pair aggregated independently
        - First (k-1) rows per pair → nulls (min_periods=k)
        - (pair, timestamp) columns preserved

    Use cases:
        - Create higher timeframe features (e.g., 5m bars from 1m bars)
        - Smooth noise with rolling aggregates
        - Generate multi-timeframe features for models

    Attributes:
        offset_window (int): Number of bars in sliding window. Must be > 0.
        ts_col (str): Timestamp column name. Default: "timestamp".
        pair_col (str): Trading pair column name. Default: "pair".
        mode (Literal["add", "replace"]): Output mode:
            - "add": Add resampled columns with prefix
            - "replace": Replace original OHLC columns
        prefix (str | None): Prefix for output columns in "add" mode.
            Default: "rs_{offset_window}m_"
        raw_data_type (RawDataType): Type of raw data. Default: SPOT.
        OFFSET_COL (str): Column name for offset tracking. Default: "resample_offset".

    Example:
        ```python
        from signalflow.core import RollingAggregator
        import polars as pl

        # Create 5-minute bars from 1-minute bars
        aggregator = RollingAggregator(
            offset_window=5,
            mode="add",
            prefix="5m_"
        )

        # Resample data
        df_resampled = aggregator.resample(spot_df)

        # Result has both 1m and 5m data
        print(df_resampled.columns)
        # ['pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
        #  '5m_open', '5m_high', '5m_low', '5m_close', '5m_volume']

        # Replace mode - output only 5m bars
        aggregator_replace = RollingAggregator(
            offset_window=5,
            mode="replace"
        )
        df_5m = aggregator_replace.resample(spot_df)

        # Add offset column for tracking
        df_with_offset = aggregator.add_offset_column(spot_df)
        print(df_with_offset["resample_offset"])  # 0, 1, 2, 3, 4, 0, 1, ...
        ```

    Note:
        Currently only supports SPOT data type (OHLCV).
        First (k-1) rows per pair will have null values for resampled columns.
        DataFrame must be sorted by (pair, timestamp) - automatic sorting applied.

    See Also:
        FeatureExtractor: For extracting features from resampled data.
    """

    offset_window: int = 1
    ts_col: str = "timestamp"
    pair_col: str = "pair"
    mode: Literal["add", "replace"] = "replace"
    prefix: str | None = None
    raw_data_type: RawDataType | str = RawDataType.SPOT

    OFFSET_COL: str = "resample_offset"

    @property
    def out_prefix(self) -> str:
        """Get output prefix for resampled columns.

        Returns:
            str: Prefix for output columns. Uses custom prefix if provided,
                otherwise defaults to "rs_{offset_window}m_".

        Example:
            ```python
            # Default prefix
            agg = RollingAggregator(offset_window=5)
            assert agg.out_prefix == "rs_5m_"

            # Custom prefix
            agg = RollingAggregator(offset_window=5, prefix="5min_")
            assert agg.out_prefix == "5min_"
            ```
        """
        return self.prefix if self.prefix is not None else f"rs_{self.offset_window}m_"

    def _validate_base(self, df: pl.DataFrame) -> None:
        """Validate base requirements for DataFrame.

        Args:
            df (pl.DataFrame): DataFrame to validate.

        Raises:
            ValueError: If offset_window <= 0.
            ValueError: If required columns (ts_col, pair_col) are missing.
        """
        if self.offset_window <= 0:
            raise ValueError(f"offset_window must be > 0, got {self.offset_window}")
        if self.ts_col not in df.columns:
            raise ValueError(f"Missing '{self.ts_col}' column")
        if self.pair_col not in df.columns:
            raise ValueError(f"Missing '{self.pair_col}' column")

    def add_offset_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add offset column for tracking position within window.

        Computes: timestamp.minute() % offset_window

        Useful for:
            - Debugging resampling logic
            - Identifying bar position within window (0, 1, 2, ..., k-1)
            - Aligning multiple dataframes

        Args:
            df (pl.DataFrame): Input DataFrame with timestamp column.

        Returns:
            pl.DataFrame: DataFrame with added offset column.

        Raises:
            ValueError: If validation fails (missing columns, invalid window).

        Example:
            ```python
            # Add offset for 5-minute windows
            agg = RollingAggregator(offset_window=5)
            df_with_offset = agg.add_offset_column(spot_df)

            # Offset cycles: 0, 1, 2, 3, 4, 0, 1, 2, ...
            print(df_with_offset["resample_offset"].to_list()[:10])
            # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

            # Filter by specific offset
            df_offset_0 = df_with_offset.filter(
                pl.col("resample_offset") == 0
            )
            ```
        """
        self._validate_base(df)

        return df.with_columns(
            (pl.col(self.ts_col).dt.minute() % pl.lit(self.offset_window)).cast(pl.Int64).alias(self.OFFSET_COL)
        )

    def get_last_offset(self, df: pl.DataFrame) -> int:
        """Get offset value of last timestamp in DataFrame.

        Useful for:
            - Determining current position in resampling window
            - Synchronizing multiple dataframes
            - Tracking resampling state

        Args:
            df (pl.DataFrame): Input DataFrame with timestamp column.

        Returns:
            int: Offset value (0 to offset_window-1).

        Raises:
            ValueError: If DataFrame is empty or validation fails.

        Example:
            ```python
            agg = RollingAggregator(offset_window=5)
            
            # Check current offset
            last_offset = agg.get_last_offset(spot_df)
            print(f"Current offset: {last_offset}")  # 0-4

            # Wait for window completion
            if last_offset == 4:
                print("Window complete, ready to resample")
            ```
        """
        self._validate_base(df)
        if df.is_empty():
            raise ValueError("Empty dataframe")

        last_ts = df.select(pl.col(self.ts_col).max()).item()
        return int(last_ts.minute % self.offset_window)

    def _spot_validate(self, cols: list[str]) -> None:
        """Validate SPOT data requirements.

        Args:
            cols (list[str]): Column names in DataFrame.

        Raises:
            ValueError: If required OHLC columns are missing.
        """
        required = {"open", "high", "low", "close"}
        missing = required - set(cols)
        if missing:
            raise ValueError(f"spot resample requires columns {sorted(required)}; missing {sorted(missing)}")

    def resample(self, df: pl.DataFrame) -> pl.DataFrame:
        """Resample DataFrame using rolling window aggregation.

        Aggregation rules for SPOT data:
            - open: Value from (k-1) bars ago (shifted)
            - high: Maximum over window
            - low: Minimum over window
            - close: Current value (no aggregation)
            - volume: Sum over window (if present)
            - trades: Sum over window (if present)

        Processing:
            1. Sort by (pair, timestamp)
            2. Add offset column if needed
            3. Apply rolling aggregations per pair
            4. Return length-preserving result

        Args:
            df (pl.DataFrame): Input DataFrame with OHLCV data.

        Returns:
            pl.DataFrame: Resampled DataFrame. Length equals input length.

        Raises:
            NotImplementedError: If raw_data_type is not SPOT.
            ValueError: If required columns missing or output length mismatch.

        Example:
            ```python
            import polars as pl
            from datetime import datetime, timedelta

            # Create 1-minute bars
            df = pl.DataFrame({
                "pair": ["BTCUSDT"] * 10,
                "timestamp": [
                    datetime(2024, 1, 1, 10, i) 
                    for i in range(10)
                ],
                "open": [45000 + i*10 for i in range(10)],
                "high": [45100 + i*10 for i in range(10)],
                "low": [44900 + i*10 for i in range(10)],
                "close": [45050 + i*10 for i in range(10)],
                "volume": [100.0] * 10
            })

            # Create 5-minute bars (add mode)
            agg = RollingAggregator(offset_window=5, mode="add")
            df_resampled = agg.resample(df)

            # First 4 rows have nulls for 5m columns
            print(df_resampled.filter(pl.col("5m_open").is_null()).height)  # 4

            # From row 5 onwards, 5m data available
            print(df_resampled[5:])
            # 5m_open = open from 5 bars ago
            # 5m_high = max(high) over last 5 bars
            # 5m_low = min(low) over last 5 bars
            # 5m_close = current close
            # 5m_volume = sum(volume) over last 5 bars

            # Replace mode - output only resampled columns
            agg_replace = RollingAggregator(offset_window=5, mode="replace")
            df_5m = agg_replace.resample(df)
            print(df_5m.columns)  # ['pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            # But values are 5-minute aggregates
            ```

        Note:
            First (k-1) rows per pair will have null values (min_periods=k).
            Input DataFrame is automatically sorted by (pair, timestamp).
            Volume and trades columns are optional but recommended.
        """
        if self.raw_data_type != RawDataType.SPOT:
            raise NotImplementedError("Currently resample() implemented for data_type='spot' only")

        self._validate_base(df)
        self._spot_validate(df.columns)

        df0 = df.sort([self.pair_col, self.ts_col])

        if self.OFFSET_COL not in df0.columns:
            df0 = self.add_offset_column(df0)

        k = int(self.offset_window)
        pfx = self.out_prefix
        over = [self.pair_col]

        rs_open = pl.col("open").shift(k - 1).over(over)
        rs_high = pl.col("high").rolling_max(window_size=k, min_periods=k).over(over)
        rs_low = pl.col("low").rolling_min(window_size=k, min_periods=k).over(over)
        rs_close = pl.col("close")

        has_volume = "volume" in df0.columns
        has_trades = "trades" in df0.columns

        if self.mode == "add":
            exprs: list[pl.Expr] = [
                rs_open.alias(f"{pfx}open"),
                rs_high.alias(f"{pfx}high"),
                rs_low.alias(f"{pfx}low"),
                rs_close.alias(f"{pfx}close"),
            ]
            if has_volume:
                exprs.append(
                    pl.col("volume")
                    .rolling_sum(window_size=k, min_periods=k)
                    .over(over)
                    .alias(f"{pfx}volume")
                )
            if has_trades:
                exprs.append(
                    pl.col("trades")
                    .rolling_sum(window_size=k, min_periods=k)
                    .over(over)
                    .alias(f"{pfx}trades")
                )
            out = df0.with_columns(exprs)

        elif self.mode == "replace":
            exprs2: list[pl.Expr] = [
                rs_open.alias("open"),
                rs_high.alias("high"),
                rs_low.alias("low"),
                rs_close.alias("close"),
            ]
            if has_volume:
                exprs2.append(
                    pl.col("volume").rolling_sum(window_size=k, min_periods=k).over(over).alias("volume")
                )
            if has_trades:
                exprs2.append(
                    pl.col("trades").rolling_sum(window_size=k, min_periods=k).over(over).alias("trades")
                )
            out = df0.with_columns(exprs2)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if out.height != df.height:
            raise ValueError(f"resample(pl): len(out)={out.height} != len(in)={df.height}")

        return out