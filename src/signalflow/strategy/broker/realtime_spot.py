#TODO: Implement
from signalflow.strategy.broker.base import Broker
from signalflow.core.decorators import sf_component
from dataclasses import dataclass

@dataclass
@sf_component(name="live/spot")
class RealtimeSpotBroker(Broker):
    """
    Live broker for spot trading.
    """
    pass
