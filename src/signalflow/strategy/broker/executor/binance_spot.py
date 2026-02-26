from dataclasses import dataclass

from signalflow.core import executor
from signalflow.strategy.broker.executor.base import OrderExecutor


@dataclass
@executor("binance/spot")
class BinanceSpotExecutor(OrderExecutor):
    """
    Binance executor for live trading.
    """

    pass
