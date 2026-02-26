"""SignalFlow Core Module.

Provides fundamental building blocks for SignalFlow trading framework:
- Containers: RawData, Signals, Position, Trade, Portfolio, Order, OrderFill
- Enums: SignalType, SfComponentType, PositionType, etc.
- Registry: Component registration and discovery
- Decorators: Semantic decorators for component registration
  - @sf.detector, @sf.feature, @sf.validator, @sf.labeler
  - @sf.entry, @sf.exit
  - @sf.signal_metric, @sf.strategy_metric
  - @sf.alert, @sf.data_source, @sf.data_store, @sf.executor, @sf.risk
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
from signalflow.core.decorators import (
    # Semantic decorators (new API)
    alert,
    data_source,
    data_store,
    detector,
    entry,
    executor,
    exit,
    feature,
    labeler,
    register,
    risk,
    signal_metric,
    strategy_metric,
    strategy_store,
    validator,
    # Legacy (deprecated)
    sf_component,
)
from signalflow.core.enums import (
    DataFrameType,
    ExitPriority,
    PositionType,
    RawDataType,
    SfComponentType,
    SignalCategory,
    SignalType,
)
from signalflow.core.registry import (
    ComponentInfo,
    SignalFlowRegistry,
    default_registry,
    get_component,
    get_component_info,
)
from signalflow.core.signal_transform import SignalsTransform

__all__ = [
    # Enums
    "DataFrameType",
    "ExitPriority",
    "PositionType",
    "RawDataType",
    "SfComponentType",
    "SignalCategory",
    "SignalType",
    # Containers
    "Order",
    "OrderFill",
    "Portfolio",
    "Position",
    "RawData",
    "RawDataLazy",
    "RawDataView",
    "Signals",
    "StrategyState",
    "Trade",
    # Registry
    "ComponentInfo",
    "SignalFlowRegistry",
    "default_registry",
    "get_component",
    "get_component_info",
    # Semantic decorators (new API)
    "alert",
    "data_source",
    "data_store",
    "detector",
    "entry",
    "executor",
    "exit",
    "feature",
    "labeler",
    "register",
    "risk",
    "signal_metric",
    "strategy_metric",
    "strategy_store",
    "validator",
    # Legacy decorator (deprecated)
    "sf_component",
    # Other
    "SfTorchModuleMixin",
    "SignalsTransform",
]
