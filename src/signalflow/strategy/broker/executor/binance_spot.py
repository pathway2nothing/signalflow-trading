from signalflow.strategy.broker.executor.base import OrderExecutor
from signalflow.core.decorators import sf_component
from dataclasses import dataclass


@dataclass
@sf_component(name="binance/spot")
class BinanceSpotExecutor(OrderExecutor):
    """
    Binance executor for live trading.
    """

    pass
