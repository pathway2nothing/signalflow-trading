from enum import Enum


class SignalType(str, Enum):
    """Enumeration of signal types.

    Represents the direction of a trading signal detected by signal detectors.

    Values:
        NONE: No signal detected or neutral state.
        RISE: Bullish signal indicating potential price increase.
        FALL: Bearish signal indicating potential price decrease.

    Example:
        ```python
        from signalflow.core.enums import SignalType

        # Check signal type
        if signal_type == SignalType.RISE:
            print("Bullish signal detected")
        elif signal_type == SignalType.FALL:
            print("Bearish signal detected")
        else:
            print("No signal")

        # Use in DataFrame
        import polars as pl
        signals_df = pl.DataFrame({
            "pair": ["BTCUSDT"],
            "timestamp": [datetime.now()],
            "signal_type": [SignalType.RISE.value]
        })

        # Compare with enum
        is_rise = signals_df.filter(
            pl.col("signal_type") == SignalType.RISE.value
        )
        ```

    Note:
        Stored as string values in DataFrames for serialization.
        Use .value to get string representation.
    """
    NONE = "none"
    RISE = "rise"
    FALL = "fall"


class PositionType(str, Enum):
    """Enumeration of position types.

    Represents the direction of a trading position.

    Values:
        LONG: Long position (profit from price increase).
        SHORT: Short position (profit from price decrease).

    Example:
        ```python
        from signalflow.core import Position, PositionType

        # Create long position
        long_position = Position(
            pair="BTCUSDT",
            position_type=PositionType.LONG,
            entry_price=45000.0,
            qty=0.5
        )

        # Check position type
        if position.position_type == PositionType.LONG:
            print("Long position")
            assert position.side_sign == 1.0
        else:
            print("Short position")
            assert position.side_sign == -1.0

        # Store in DataFrame
        positions_df = pl.DataFrame({
            "pair": ["BTCUSDT"],
            "position_type": [PositionType.LONG.value],
            "qty": [0.5]
        })
        ```

    Note:
        Currently only LONG positions are fully implemented.
        SHORT positions planned for future versions.
    """
    LONG = "long"
    SHORT = "short"


class SfComponentType(str, Enum):
    """Enumeration of SignalFlow component types.

    Defines all component types that can be registered in the component registry.
    Used by sf_component decorator and SignalFlowRegistry for type-safe registration.

    Component categories:
        - Data: Raw data loading and storage
        - Feature: Feature extraction
        - Signals: Signal detection, transformation, labeling, validation
        - Strategy: Execution, rules, metrics

    Values:
        RAW_DATA_STORE: Raw data storage backends (e.g., DuckDB, Parquet).
        RAW_DATA_SOURCE: Raw data sources (e.g., Binance API).
        RAW_DATA_LOADER: Raw data loaders combining source + store.
        FEATURE_EXTRACTOR: Feature extraction classes (e.g., RSI, SMA).
        SIGNALS_TRANSFORM: Signal transformation functions.
        LABELER: Signal labeling strategies (e.g., triple barrier).
        DETECTOR: Signal detection algorithms (e.g., SMA cross).
        VALIDATOR: Signal validation models.
        TORCH_MODULE: PyTorch neural network modules.
        VALIDATOR_MODEL: Pre-trained validator models.
        STRATEGY_STORE: Strategy state persistence backends.
        STRATEGY_RUNNER: Backtest/live runner implementations.
        STRATEGY_BROKER: Order management and position tracking.
        STRATEGY_EXECUTOR: Order execution engines (backtest/live).
        STRATEGY_EXIT_RULE: Position exit rules (e.g., take profit, stop loss).
        STRATEGY_ENTRY_RULE: Position entry rules (e.g., fixed size).
        STRATEGY_METRIC: Strategy performance metrics.

    Example:
        ```python
        from signalflow.core import sf_component
        from signalflow.core.enums import SfComponentType
        from signalflow.detector import SignalDetector

        # Register detector
        @sf_component(name="my_detector")
        class MyDetector(SignalDetector):
            component_type = SfComponentType.DETECTOR
            # ... implementation

        # Register extractor
        @sf_component(name="my_feature")
        class MyExtractor(FeatureExtractor):
            component_type = SfComponentType.FEATURE_EXTRACTOR
            # ... implementation

        # Register exit rule
        @sf_component(name="my_exit")
        class MyExit(ExitRule):
            component_type = SfComponentType.STRATEGY_EXIT_RULE
            # ... implementation

        # Use in registry
        from signalflow.core.registry import default_registry

        detector = default_registry.create(
            SfComponentType.DETECTOR,
            "my_detector"
        )
        ```

    Note:
        All registered components must have component_type class attribute.
        Component types are organized hierarchically (category/subcategory).
    """
    RAW_DATA_STORE = "data/store"
    RAW_DATA_SOURCE = "data/source"
    RAW_DATA_LOADER = "data/loader"

    FEATURE = "feature"
    SIGNALS_TRANSFORM = "signals/transform"
    SIGNAL_METRIC = "signals/metric"
    LABELER = "signals/labeler"
    DETECTOR = "signals/detector"
    VALIDATOR = "signals/validator"
    TORCH_MODULE = "torch_module"
    VALIDATOR_MODEL = "signals/validator/model"

    STRATEGY_STORE = "strategy/store"
    STRATEGY_RUNNER = "strategy/runner"
    STRATEGY_BROKER = "strategy/broker"
    STRATEGY_EXECUTOR = "strategy/executor"
    STRATEGY_EXIT_RULE = "strategy/exit"
    STRATEGY_ENTRY_RULE = "strategy/entry"
    STRATEGY_METRIC = "strategy/metric"


class DataFrameType(str, Enum):
    """Supported DataFrame backends.

    Specifies which DataFrame library to use for data processing.
    Used by FeatureExtractor and other components to determine input/output format.

    Values:
        POLARS: Polars DataFrame (faster, modern).
        PANDAS: Pandas DataFrame (legacy compatibility).

    Example:
        ```python
        from signalflow.core.enums import DataFrameType
        from signalflow.feature import FeatureExtractor

        # Polars-based extractor
        class MyExtractor(FeatureExtractor):
            df_type = DataFrameType.POLARS
            
            def extract(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    pl.col("close").rolling_mean(20).alias("sma_20")
                )

        # Pandas-based extractor
        class LegacyExtractor(FeatureExtractor):
            df_type = DataFrameType.PANDAS
            
            def extract(self, df: pd.DataFrame) -> pd.DataFrame:
                df["sma_20"] = df["close"].rolling(20).mean()
                return df

        # Use in RawDataView
        from signalflow.core import RawDataView

        view = RawDataView(raw=raw_data)
        
        # Get data in required format
        df_polars = view.get_data("spot", DataFrameType.POLARS)
        df_pandas = view.get_data("spot", DataFrameType.PANDAS)
        ```

    Note:
        New code should prefer POLARS for better performance.
        PANDAS supported for backward compatibility and legacy libraries.
    """
    POLARS = "polars"
    PANDAS = "pandas"

class RawDataType(str, Enum):
    """Supported raw data types.

    Defines types of market data that can be loaded and processed.

    Values:
        SPOT: Spot trading data (OHLCV).

    Example:
        ```python
        from signalflow.core.enums import RawDataType

        # Load spot data
        loader = BinanceLoader(
            pairs=["BTCUSDT", "ETHUSDT"],
            data_type=RawDataType.SPOT
        )

        raw_data = loader.load(
            datetime_start=datetime(2024, 1, 1),
            datetime_end=datetime(2024, 12, 31)
        )

        # Access spot data
        spot_df = raw_data[RawDataType.SPOT.value]

        # Check data type
        if raw_data_type == RawDataType.SPOT:
            print("Processing spot data")
        ```

    Note:
        Future versions will add:
        - FUTURES: Futures trading data
        - PERPETUAL: Perpetual swaps data
        - LOB: Limit order book data
    """
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    
    @property
    def columns(self) -> set[str]:
        """Columns guaranteed to be present."""
        base = {"pair", "timestamp", "open", "high", "low", "close", "volume"}
        
        if self == RawDataType.FUTURES:
            return base | {"open_interest"}
        elif self == RawDataType.PERPETUAL:
            return base | {"funding_rate", "open_interest"}
        
        return base

