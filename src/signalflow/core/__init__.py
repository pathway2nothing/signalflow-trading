"""SignalFlow Core Module.

Provides fundamental building blocks for SignalFlow trading framework:
- Containers: RawData, Signals, Position, Trade, Portfolio, Order, OrderFill
- Enums: SignalType, SfComponentType, PositionType, etc.
- Registry: Component registration and discovery
- Decorators: @sf_component for automatic registration
- Transforms: SignalsTransform protocol
"""

from signalflow.core.base_mixin import SfTorchModuleMixin
from signalflow.core.containers import (
    Order,
    OrderFill,
    Portfolio,
    Position,
    RawData,
    RawDataLazy,
    RawDataView,
    Signals,
    StrategyState,
    Trade,
)
from signalflow.core.decorators import sf_component
from signalflow.core.enums import (
    DataFrameType,
    ExitPriority,
    PositionType,
    RawDataType,
    SfComponentType,
    SignalCategory,
    SignalType,
)
from signalflow.core.registry import SignalFlowRegistry, default_registry, get_component
from signalflow.core.signal_transform import SignalsTransform

__all__ = [
    "DataFrameType",
    "ExitPriority",
    "Order",
    "OrderFill",
    "Portfolio",
    "Position",
    "PositionType",
    "RawData",
    "RawDataLazy",
    "RawDataType",
    "RawDataView",
    "SfComponentType",
    "SfTorchModuleMixin",
    "SignalCategory",
    "SignalFlowRegistry",
    "SignalType",
    "Signals",
    "SignalsTransform",
    "StrategyState",
    "Trade",
    "default_registry",
    "get_component",
    "sf_component",
]
