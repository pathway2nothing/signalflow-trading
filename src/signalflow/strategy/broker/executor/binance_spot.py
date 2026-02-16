from dataclasses import dataclass

from signalflow.core.decorators import sf_component
from signalflow.strategy.broker.executor.base import OrderExecutor


@dataclass
@sf_component(name="binance/spot")
class BinanceSpotExecutor(OrderExecutor):
    """
    Binance executor for live trading.
    """

    pass
