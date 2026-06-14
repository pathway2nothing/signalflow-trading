"""StrategyModel protocol."""


from typing import Protocol, runtime_checkable

from signalflow.engine.types import Intent
from signalflow.strategy.observation import Observation


@runtime_checkable
class StrategyModel(Protocol):
    def decide(self, obs: Observation) -> list[Intent]: ...
