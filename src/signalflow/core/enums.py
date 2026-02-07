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


class ExitPriority(str, Enum):
    """Priority mode for composite exit rules.

    Defines how multiple exit rules are combined in CompositeExit.

    Values:
        FIRST_TRIGGERED: First rule to generate exit wins per position.
        HIGHEST_PRIORITY: All rules evaluated, lowest priority number wins.
        ALL_MUST_AGREE: All rules must agree on exit (intersection).

    Example:
        ```python
        from signalflow.core.enums import ExitPriority
        from signalflow.strategy.component.exit import CompositeExit

        composite = CompositeExit(
            rules=[
                (tp_sl_exit, 1),      # Priority 1 (highest)
                (trailing_exit, 2),   # Priority 2
                (time_exit, 3),       # Priority 3 (lowest)
            ],
            priority_mode=ExitPriority.HIGHEST_PRIORITY
        )
        ```
    """

    FIRST_TRIGGERED = "first_triggered"
    HIGHEST_PRIORITY = "highest_priority"
    ALL_MUST_AGREE = "all_must_agree"


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
        STRATEGY_ALERT: Strategy monitoring alerts (e.g., max drawdown, stuck positions).

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
    STRATEGY_ALERT = "strategy/alert"


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
    """Built-in raw data types.

    Defines types of market data that can be loaded and processed.
    Column definitions are stored in :class:`SignalFlowRegistry` and can be
    extended with custom types via ``default_registry.register_raw_data_type()``.

    Values:
        SPOT: Spot trading data (OHLCV).
        FUTURES: Futures trading data (OHLCV + open_interest).
        PERPETUAL: Perpetual swaps data (OHLCV + funding_rate + open_interest).

    Example:
        ```python
        from signalflow.core.enums import RawDataType

        # Built-in types
        spot_cols = RawDataType.SPOT.columns
        # {'pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}

        # Custom types - register via registry
        from signalflow.core.registry import default_registry

        default_registry.register_raw_data_type(
            name="lob",
            columns=["pair", "timestamp", "bid", "ask", "bid_size", "ask_size"],
        )
        cols = default_registry.get_raw_data_columns("lob")
        ```

    Note:
        Use ``default_registry.register_raw_data_type()`` to add custom types.
        Use ``default_registry.get_raw_data_columns(name)`` to look up columns
        for any type (built-in or custom).
    """

    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"

    @property
    def columns(self) -> set[str]:
        """Columns guaranteed to be present (looked up from registry)."""
        from signalflow.core.registry import default_registry

        return default_registry.get_raw_data_columns(self.value)
