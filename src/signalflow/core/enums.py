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
    EXIT = "strategy/exit"
    NEW_POSITION = "strategy/new-position"
    


class DataFrameType(str, Enum):
    """Supported dataframe backends."""
    POLARS = "polars"
    PANDAS = "pandas"

class RawDataType(str, Enum):
    """Supported raw data types."""
    SPOT = "spot"

