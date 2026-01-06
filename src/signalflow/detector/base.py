from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import RawDataView, Signals, SfComponentType, SignalType, RawDataType
from signalflow.feature import FeatureSet


@dataclass
class SignalDetector(ABC):
    """Base class for Polars-first signal detection.

    Provides standardized pipeline for detecting trading signals from raw data:
        1. preprocess: Extract features from raw data
        2. detect: Generate signals from features
        3. validate: Ensure data quality

    Key features:
        - Polars-native for performance
        - Automatic feature extraction via FeatureSet
        - Built-in validation (schema, duplicates, timezones)
        - Optional probability requirement
        - Keep latest signal per pair option

    Public API:
        - run(): Complete pipeline (preprocess → detect → validate)
        - preprocess(): Feature extraction (delegates to FeatureSet)
        - detect(): Signal generation (must implement)

    Attributes:
        component_type (ClassVar[SfComponentType]): Always DETECTOR for registry.
        pair_col (str): Trading pair column name. Default: "pair".
        ts_col (str): Timestamp column name. Default: "timestamp".
        raw_data_type (RawDataType): Type of raw data to process. Default: SPOT.
        feature_set (FeatureSet | None): Feature extractor. Default: None.
        require_probability (bool): Require probability column in signals. Default: False.
        keep_only_latest_per_pair (bool): Keep only latest signal per pair. Default: False.

    Example:
        ```python
        from signalflow.detector import SignalDetector
        from signalflow.core import Signals, SignalType
        import polars as pl

        class SmaCrossDetector(SignalDetector):
            '''Simple SMA crossover detector'''
            
            def __init__(self, fast_window: int = 10, slow_window: int = 20):
                super().__init__()
                # Auto-generate features
                from signalflow.feature import FeatureSet, SmaExtractor
                self.feature_set = FeatureSet([
                    SmaExtractor(window=fast_window, column="close"),
                    SmaExtractor(window=slow_window, column="close")
                ])
            
            def detect(self, features: pl.DataFrame, context=None) -> Signals:
                signals = features.with_columns([
                    # Detect crossover
                    (pl.col("sma_10") > pl.col("sma_20")).alias("is_bull"),
                    (pl.col("sma_10") < pl.col("sma_20")).alias("is_bear")
                ]).with_columns([
                    # Assign signal type
                    pl.when(pl.col("is_bull"))
                    .then(pl.lit(SignalType.RISE.value))
                    .when(pl.col("is_bear"))
                    .then(pl.lit(SignalType.FALL.value))
                    .otherwise(pl.lit(SignalType.NONE.value))
                    .alias("signal_type")
                ]).select([
                    self.pair_col,
                    self.ts_col,
                    "signal_type",
                    pl.lit(1).alias("signal")
                ])
                
                return Signals(signals)

        # Usage
        detector = SmaCrossDetector(fast_window=10, slow_window=20)
        signals = detector.run(raw_data_view)
        ```

    Note:
        Subclasses must implement detect() method.
        All DataFrames must use timezone-naive timestamps.
        Duplicate (pair, timestamp) combinations are rejected.

    See Also:
        FeatureSet: Orchestrates feature extraction.
        Signals: Container for signal output.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    raw_data_type: RawDataType = RawDataType.SPOT

    feature_set: FeatureSet | None = None

    require_probability: bool = False
    keep_only_latest_per_pair: bool = False

    def run(self, raw_data_view: RawDataView, context: dict[str, Any] | None = None) -> Signals:
        """Execute complete detection pipeline.

        Pipeline steps:
            1. preprocess: Extract features
            2. normalize: Ensure timezone-naive timestamps
            3. validate features: Check schema and duplicates
            4. detect: Generate signals
            5. validate signals: Check output quality
            6. (optional) keep latest: Filter to latest per pair

        Args:
            raw_data_view (RawDataView): View to raw market data.
            context (dict[str, Any] | None): Additional context for detection.

        Returns:
            Signals: Detected signals.

        Raises:
            TypeError: If preprocess doesn't return pl.DataFrame.
            ValueError: If features/signals fail validation.

        Example:
            ```python
            from signalflow.core import RawData, RawDataView

            # Create view
            view = RawDataView(raw=raw_data)

            # Run detection
            signals = detector.run(view)

            # With context
            signals = detector.run(view, context={"threshold": 0.7})
            ```

        Note:
            Can also be called directly: detector(raw_data_view).
            All validation errors include helpful diagnostic information.
        """
        feats = self.preprocess(raw_data_view, context=context)
        feats = self._normalize_index(feats)
        self._validate_features(feats)

        signals = self.detect(feats, context=context)
        self._validate_signals(signals)

        if self.keep_only_latest_per_pair:
            signals = self._keep_only_latest(signals)

        return signals

    __call__ = run

    def preprocess(self, raw_data_view: RawDataView, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Extract features from raw data.

        Default implementation delegates to FeatureSet. Override for custom
        feature extraction logic.

        Args:
            raw_data_view (RawDataView): View to raw market data.
            context (dict[str, Any] | None): Additional context.

        Returns:
            pl.DataFrame: Features with at minimum pair and timestamp columns.

        Raises:
            NotImplementedError: If feature_set is None and not overridden.
            TypeError: If FeatureSet doesn't return pl.DataFrame.

        Example:
            ```python
            # Default: uses FeatureSet
            features = detector.preprocess(raw_data_view)

            # Custom override
            class CustomDetector(SignalDetector):
                def preprocess(self, raw_data_view, context=None):
                    df = raw_data_view.to_polars("spot")
                    return df.with_columns([
                        pl.col("close").rolling_mean(10).alias("sma_10")
                    ])
            ```
        """
        if self.feature_set is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.preprocess is not implemented and feature_set is None"
            )
        out = self.feature_set.extract(raw_data_view, context=context)
        if not isinstance(out, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.feature_set.extract must return pl.DataFrame, got {type(out)}")
        return out

    @abstractmethod
    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals from features.

        Core detection logic - must be implemented by subclasses.

        Args:
            features (pl.DataFrame): Preprocessed features.
            context (dict[str, Any] | None): Additional context.

        Returns:
            Signals: Detected signals with columns:
                - pair (str): Trading pair
                - timestamp (datetime): Signal timestamp (timezone-naive)
                - signal_type (int): SignalType enum value
                - signal (int | float): Signal value
                - probability (float, optional): Signal probability

        Example:
            ```python
            def detect(self, features, context=None):
                # Simple threshold detector
                signals = features.filter(
                    pl.col("rsi") > 70  # Overbought
                ).with_columns([
                    pl.lit(SignalType.FALL.value).alias("signal_type"),
                    pl.lit(-1).alias("signal"),
                    pl.lit(0.8).alias("probability")
                ]).select([
                    self.pair_col,
                    self.ts_col,
                    "signal_type",
                    "signal",
                    "probability"
                ])
                
                return Signals(signals)
            ```

        Note:
            Must return Signals with at minimum: pair, timestamp, signal_type.
            Timestamps must be timezone-naive.
            No duplicate (pair, timestamp) combinations allowed.
        """
        raise NotImplementedError

    def _normalize_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize timestamps to timezone-naive.

        Args:
            df (pl.DataFrame): Input DataFrame.

        Returns:
            pl.DataFrame: DataFrame with timezone-naive timestamps.

        Raises:
            TypeError: If df is not pl.DataFrame.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame, got {type(df)}")

        if self.ts_col in df.columns:
            ts_dtype = df.schema.get(self.ts_col)
            if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
                df = df.with_columns(pl.col(self.ts_col).dt.replace_time_zone(None))
        return df

    def _validate_features(self, df: pl.DataFrame) -> None:
        """Validate feature DataFrame.

        Checks:
            - Is pl.DataFrame
            - Has required columns (pair, timestamp)
            - Timestamps are timezone-naive
            - No duplicate (pair, timestamp) combinations

        Args:
            df (pl.DataFrame): Features to validate.

        Raises:
            TypeError: If not pl.DataFrame.
            ValueError: If validation fails.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"preprocess must return polars.DataFrame, got {type(df)}")

        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Features missing required columns: {missing}")

        ts_dtype = df.schema.get(self.ts_col)
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
            raise ValueError(
                f"Features column '{self.ts_col}' must be timezone-naive, got tz={ts_dtype.time_zone}. "
                f"Use .dt.replace_time_zone(None)."
            )

        dup = (
            df.group_by([self.pair_col, self.ts_col])
            .len()
            .filter(pl.col("len") > 1)
        )
        if dup.height > 0:
            raise ValueError(
                "Features contain duplicate keys (pair,timestamp). "
                f"Examples:\n{dup.select([self.pair_col, self.ts_col]).head(10)}"
            )

    def _validate_signals(self, signals: Signals) -> None:
        """Validate signal output.

        Checks:
            - Is Signals instance with pl.DataFrame value
            - Has required columns (pair, timestamp, signal_type)
            - signal_type values are valid SignalType enums
            - Timestamps are timezone-naive
            - No duplicate (pair, timestamp) combinations
            - (optional) Has probability column if required

        Args:
            signals (Signals): Signals to validate.

        Raises:
            TypeError: If not Signals or value not pl.DataFrame.
            ValueError: If validation fails.
        """
        if not isinstance(signals, Signals):
            raise TypeError(f"detect must return Signals, got {type(signals)}")

        s = signals.value
        if not isinstance(s, pl.DataFrame):
            raise TypeError(f"Signals.value must be polars.DataFrame, got {type(s)}")

        required = {self.pair_col, self.ts_col, "signal_type"}
        missing = sorted(required - set(s.columns))
        if missing:
            raise ValueError(f"Signals missing required columns: {missing}")

        allowed = {t.value for t in SignalType}
        bad = (
            s.select(pl.col("signal_type"))
            .unique()
            .filter(~pl.col("signal_type").is_in(list(allowed)))
        )
        if bad.height > 0:
            raise ValueError(
                f"Signals contain unknown signal_type values: {bad.get_column('signal_type').to_list()}"
            )

        if self.require_probability and "probability" not in s.columns:
            raise ValueError("Signals must contain 'probability' column (require_probability=True)")

        ts_dtype = s.schema.get(self.ts_col)
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
            raise ValueError(f"Signals column '{self.ts_col}' must be timezone-naive, got tz={ts_dtype.time_zone}.")

        # optional: hard guarantee no duplicates in signals
        dup = (
            s.group_by([self.pair_col, self.ts_col])
            .len()
            .filter(pl.col("len") > 1)
        )
        if dup.height > 0:
            raise ValueError(
                "Signals contain duplicate keys (pair,timestamp). "
                f"Examples:\n{dup.select([self.pair_col, self.ts_col]).head(10)}"
            )

    def _keep_only_latest(self, signals: Signals) -> Signals:
        """Keep only latest signal per pair.

        Useful for strategies that only trade most recent signal.

        Args:
            signals (Signals): Input signals.

        Returns:
            Signals: Filtered signals with one per pair.
        """
        s = signals.value
        out = (
            s.sort([self.pair_col, self.ts_col])
            .group_by(self.pair_col, maintain_order=True)
            .tail(1)
            .sort([self.pair_col, self.ts_col])
        )
        return Signals(out)