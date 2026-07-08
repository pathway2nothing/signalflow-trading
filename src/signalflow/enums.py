"""Core enums and signal constants."""

from enum import StrEnum


class Signal(StrEnum):
    """Discrete detector output."""

    RISE = "rise"
    FALL = "fall"
    NONE = "none"


RISE: str = Signal.RISE.value
FALL: str = Signal.FALL.value
NONE: str = Signal.NONE.value

SIGNAL_COL = "signal"
"""Reserved column name a detector writes its discrete signal into."""

RESERVED_COLUMNS = frozenset(
    {"pair", "ts", "open", "high", "low", "close", "volume", "signal", "label", "_w", "weight"}
)
"""Column names never treated as model features (OHLCV keys, signal, label, and sample weights)."""


class Side(StrEnum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order type understood by brokers."""

    MARKET = "market"
    LIMIT = "limit"


class PositionSide(StrEnum):
    """Direction of an open position. Spot-first: LONG is the 1.0 default."""

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PositionSide.LONG else -1


class IntentKind(StrEnum):
    """What a strategy proposes for a pair (a proposal, not an order)."""

    OPEN = "open"
    CLOSE = "close"
    RESIZE = "resize"


class RunMode(StrEnum):
    """Execution mode of a Run (backtest, paper, live, or quicktest triage)."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    QUICKTEST = "quicktest"


class Provenance(StrEnum):
    """How forecast columns in a frame were produced (full vs out-of-fold)."""

    FULL = "full"
    OOS = "oos"


class SignalCategory(StrEnum):
    """High-level signal category a detector/labeler produces."""

    PRICE_DIRECTION = "price_direction"
    PRICE_STRUCTURE = "price_structure"
    TREND_MOMENTUM = "trend_momentum"
    VOLATILITY = "volatility"
    VOLUME_LIQUIDITY = "volume_liquidity"
    MARKET_WIDE = "market_wide"
    ANOMALY = "anomaly"


class RawDataType(StrEnum):
    """Market-data shape a transform/labeler expects (spot-first; perps later)."""

    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"


class ComponentType(StrEnum):
    """The registry component types."""

    SOURCE = "source"
    TRANSFORM = "transform"
    MODEL = "model"
    STRATEGY = "strategy"
    SAMPLER = "sampler"
    BROKER = "broker"
    METRIC = "metric"
    TARGET = "target"
