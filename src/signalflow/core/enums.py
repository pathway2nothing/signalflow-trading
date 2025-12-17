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
    FEATURE_PIPELINE = "feature/pipeline"
    DETECTOR = "detector"
    VALIDATOR = "validator"
    EXIT = "exit"


class DataFrameType(str, Enum):
    """Supported dataframe backends."""
    POLARS = "polars"
    PANDAS = "pandas"
