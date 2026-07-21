"""Engine - the one event-sourced execution core."""

from signalflow.engine.types import Fill, PortfolioSnapshot, Position, cross_rate, parse_pair
from signalflow.enums import Side


class Engine:
    """Folds fills into balances + positions and values the book in ``target``."""

    def __init__(self, capital: dict[str, float] | float, target: str = "USDT", quote: str = "USDT"):
        if isinstance(capital, (int, float)):
            capital = {quote: float(capital)}
        self.initial_capital: dict[str, float] = dict(capital)
        self.target = target
        self.quote = quote
        self.reset()

    def reset(self) -> None:
        self.balances: dict[str, float] = dict(self.initial_capital)
        self.positions: dict[str, Position] = {}
        self.event_log: list[Fill] = []
        self.order_log: list = []
        self.marks: dict[str, float] = {}

    def record_order(self, event) -> None:
        """Append an OrderEvent to the parallel audit/recovery log (never mutates the book)."""
        self.order_log.append(event)

    def apply(self, fills: list[Fill]) -> None:
        for f in fills:
            self._apply_one(f, record=True)

    def _apply_one(self, fill: Fill, record: bool) -> None:
        base, quote = parse_pair(fill.pair)
        self.balances.setdefault(base, 0.0)
        self.balances.setdefault(quote, 0.0)
        pos = self.positions.get(fill.pair) or Position(pair=fill.pair, opened_ts=fill.ts)

        if fill.side == Side.BUY:
            self.balances[quote] -= fill.notional
            self.balances[base] += fill.qty
            new_qty = pos.qty + fill.qty
            pos.avg_price = (pos.avg_price * pos.qty + fill.price * fill.qty) / new_qty if new_qty > 0 else 0.0
            pos.qty = new_qty
            if pos.opened_ts is None:
                pos.opened_ts = fill.ts
        else:
            self.balances[base] -= fill.qty
            self.balances[quote] += fill.notional
            pos.qty -= fill.qty
            if pos.qty <= 1e-12:
                pos.qty = 0.0
                pos.avg_price = 0.0
                pos.opened_ts = None

        self.balances.setdefault(fill.fee_asset, 0.0)
        self.balances[fill.fee_asset] -= fill.fee
        self.marks[fill.pair] = fill.price

        if pos.qty > 0:
            self.positions[fill.pair] = pos
        else:
            self.positions.pop(fill.pair, None)
        if record:
            self.event_log.append(fill)

    def equity(self, prices: dict[str, float]) -> float:
        """Value the book in ``target``, carrying forward the last-known price for
        any held asset missing from ``prices`` (gappy/staggered multi-asset data)."""
        self.marks.update(prices)
        total = 0.0
        for asset, amount in self.balances.items():
            if abs(amount) < 1e-15:
                continue
            total += amount * self._rate(asset)
        return total

    def _rate(self, asset: str) -> float:
        try:
            return cross_rate(asset, self.target, self.marks)
        except KeyError:
            pos = self.positions.get(f"{asset}{self.target}")
            return pos.avg_price if pos is not None and pos.avg_price > 0 else 0.0

    def snapshot(self, ts, prices: dict[str, float]) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            ts=ts,
            target=self.target,
            balances=dict(self.balances),
            positions={k: Position(**vars(v)) for k, v in self.positions.items()},
            equity=self.equity(prices),
            prices=dict(prices),
        )

    @classmethod
    def fold(cls, fills: list[Fill], capital: dict[str, float] | float, target: str = "USDT") -> "Engine":
        eng = cls(capital, target=target)
        for f in fills:
            eng._apply_one(f, record=True)
        return eng
