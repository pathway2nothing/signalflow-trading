"""Engine value types - intents, orders, fills, positions."""


from dataclasses import dataclass, field

from signalflow.enums import IntentKind, OrderType, Side

DEFAULT_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "USD", "BTC", "ETH")


def cross_rate(base: str, quote: str, prices: dict[str, float]) -> float:
    """Price of ``base`` in ``quote`` given a pair→close map."""
    if base == quote:
        return 1.0
    direct = prices.get(f"{base}{quote}")
    if direct is not None:
        return direct
    inverse = prices.get(f"{quote}{base}")
    if inverse:
        return 1.0 / inverse
    raise KeyError(f"no cross rate {base}->{quote} in prices")


def parse_pair(pair: str, quotes: tuple[str, ...] = DEFAULT_QUOTES) -> tuple[str, str]:
    """Split ``BTCUSDT`` → (``BTC``, ``USDT``) by known quote suffix."""
    for q in quotes:
        if pair.endswith(q) and len(pair) > len(q):
            return pair[: -len(q)], q

    return pair[:-4], pair[-4:]


@dataclass
class Intent:
    """A strategy proposal for a pair (not yet an order)."""

    pair: str
    kind: IntentKind
    side: Side
    qty: float | None = None
    notional: float | None = None
    reason: str = ""


@dataclass
class Order:
    """A concrete order handed to a broker."""

    pair: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    ts: object = None
    reason: str = ""


@dataclass
class Fill:
    """An executed order - the single source of truth the Engine folds."""

    pair: str
    ts: object
    side: Side
    qty: float
    price: float
    fee: float = 0.0
    fee_asset: str = "USDT"

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    """A long (spot) position in a pair, with cost basis derived from fills."""

    pair: str
    qty: float = 0.0
    avg_price: float = 0.0
    opened_ts: object = None

    def value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.qty

    def return_pct(self, price: float) -> float:
        return (price / self.avg_price - 1.0) if self.avg_price else 0.0


@dataclass
class PortfolioSnapshot:
    """Read-only view passed into strategy/risk each bar."""

    ts: object
    target: str
    balances: dict[str, float]
    positions: dict[str, Position]
    equity: float
    prices: dict[str, float] = field(default_factory=dict)
