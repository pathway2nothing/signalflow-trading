from enum import Enum


class SignalType(Enum):
    """Enumeration of signal types."""
    NONE = 0
    RISE = 1
    FALL = 2


class PositionType(Enum):
    """Enumeration of signal types"""
    LONG = 0
    SHORT = 1


class SfComponentType(str, Enum):
    DETECTOR = "detector"
    VALIDATOR = "validator"
    EXIT = "exit"
