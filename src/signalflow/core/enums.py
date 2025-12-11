from enum import Enum

class SignalType(Enum):
    NONE = 0
    RISE = 1
    FALL = 2

class PositionType(Enum):
    LONG = 0
    SHORT = 1