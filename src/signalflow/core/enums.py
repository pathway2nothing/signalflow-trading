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
    FEATURE_EXTRACTOR = "feature/extractor"
    LABELER = "labeler"
    DETECTOR = "detector"
    VALIDATOR = "validator"
    TORCH_MODULE = "torch_module"
    MODEL = "model"


    STRATEGY_STORE = "strategy/store"
    STRATEGY_RUNNER = "strategy/runner"
    STRATEGY_EXECUTOR = "strategy/executor"

    STRATEGY_EXIT_RULE = "strategy/exit"
    STRATEGY_ENTRY_RULE = "strategy/new-position"
    STRATEGY_METRIC = "strategy/metric"


class DataFrameType(str, Enum):
    """Supported dataframe backends."""
    POLARS = "polars"
    PANDAS = "pandas"

class RawDataType(str, Enum):
    """Supported raw data types."""
    SPOT = "spot"

