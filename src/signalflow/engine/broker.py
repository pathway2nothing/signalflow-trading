"""Brokers - turn orders into fills."""


from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from signalflow.decorators import broker
from signalflow.engine.types import Fill, Order
from signalflow.enums import Side


@runtime_checkable
class Broker(Protocol):
    def execute(self, orders: list[Order], bar) -> list[Fill]: ...


@broker("sim")
@dataclass
class SimBroker(Broker):
    """Simulated fills with flat fee + slippage; next-available close pricing."""

    fee_rate: float = 0.001
    slippage: float = 0.0005
    quote: str = "USDT"

    def execute(self, orders: list[Order], bar) -> list[Fill]:
        fills: list[Fill] = []
        for o in orders:
            price = bar.prices.get(o.pair)
            if price is None or o.qty <= 0:
                continue
            exec_price = price * (1 + self.slippage) if o.side == Side.BUY else price * (1 - self.slippage)
            fee = o.qty * exec_price * self.fee_rate
            fills.append(
                Fill(
                    pair=o.pair,
                    ts=bar.ts,
                    side=o.side,
                    qty=o.qty,
                    price=exec_price,
                    fee=fee,
                    fee_asset=self.quote,
                )
            )
        return fills


@broker("exchange")
@dataclass
class ExchangeBroker(Broker):
    """Base for live venue brokers. Subclass and implement ``execute``."""

    quote: str = "USDT"

    def execute(self, orders: list[Order], bar) -> list[Fill]:
        raise NotImplementedError(
            "ExchangeBroker is abstract; use SimBroker for backtest/paper or a "
            "configured live venue client (e.g. BinanceBroker) for armed live trading."
        )
