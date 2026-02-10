"""SignalFlow Core Module.

Provides fundamental building blocks for SignalFlow trading framework:
- Containers: RawData, Signals, Position, Trade, Portfolio, Order, OrderFill
- Enums: SignalType, SfComponentType, PositionType, etc.
- Registry: Component registration and discovery
- Decorators: @sf_component for automatic registration
- Transforms: SignalsTransform protocol
"""

from signalflow.core.containers import (
    RawData,
    Signals,
    RawDataView,
    Position,
    Trade,
    Portfolio,
    StrategyState,
    Order,
    OrderFill,
)
from signalflow.core.enums import (
    SignalType,
    SignalCategory,
    PositionType,
    SfComponentType,
    DataFrameType,
    RawDataType,
    ExitPriority,
)
from signalflow.core.decorators import sf_component
from signalflow.core.registry import default_registry, SignalFlowRegistry, get_component
from signalflow.core.signal_transform import SignalsTransform
from signalflow.core.base_mixin import SfTorchModuleMixin

__all__ = [
    "RawData",
    "Signals",
    "RawDataView",
    "Position",
    "Trade",
    "Order",
    "OrderFill",
    "Portfolio",
    "StrategyState",
    "SignalType",
    "SignalCategory",
    "PositionType",
    "SfComponentType",
    "DataFrameType",
    "RawDataType",
    "ExitPriority",
    "sf_component",
    "default_registry",
    "SignalFlowRegistry",
    "get_component",
    "SignalsTransform",
    "SfTorchModuleMixin",
]
