from enum import Enum


class SignalType(str, Enum):
    """Enumeration of signal types."""
    NONE = "none"
    RISE = "rise"
    FALL = "fall"


class PositionType(str, Enum):
    """Enumeration of signal types"""
    LONG = "long"
    SHORT = "short"


class SfComponentType(str, Enum):
    RAW_DATA_STORE = "data/store"
    RAW_DATA_SOURCE = "data/source"
    RAW_DATA_LOADER = "data/loader"

    FEATURE_EXTRACTOR = "feature/extractor"

    SIGNALS_TRANSFORM = "signals/transform"
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
    """Supported dataframe backends."""
    POLARS = "polars"
    PANDAS = "pandas"

class RawDataType(str, Enum):
    """Supported raw data types."""
    SPOT = "spot"

