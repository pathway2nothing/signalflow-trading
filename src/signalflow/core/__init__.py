"""SignalFlow Core Module.

Provides fundamental building blocks for SignalFlow trading framework:
- Containers: RawData, Signals, Position, Trade, Portfolio, Order, OrderFill
- Enums: SignalType, SfComponentType, PositionType, etc.
- Registry: Component registration and discovery
- Decorators: Semantic decorators for component registration
  - @sf.detector, @sf.feature, @sf.validator, @sf.labeler
  - @sf.entry, @sf.exit
  - @sf.signal_feature, @sf.signal_metric, @sf.strategy_metric
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
    # Legacy (deprecated)
    sf_component,
    signal_feature,
    signal_metric,
    strategy_metric,
    strategy_store,
    validator,
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
from signalflow.core.eventlog import (
    CashPolicy,
    apply_fill,
    fold,
    portfolios_match,
    replay_state,
)
from signalflow.core.registry import (
    ComponentInfo,
    SignalFlowRegistry,
    default_registry,
    get_component,
    get_component_info,
)
from signalflow.core.signal_transform import SignalsTransform
from signalflow.core.warmup import (
    assert_warmup_consistency,
    required_warmup_bars,
    warmup_bars_of,
)

__all__ = [
    # Event log (source of truth)
    "CashPolicy",
    # Registry
    "ComponentInfo",
    # Enums
    "DataFrameType",
    "ExitPriority",
    # Containers
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
    # Other
    "SfTorchModuleMixin",
    "SignalCategory",
    "SignalFlowRegistry",
    "SignalType",
    "Signals",
    "SignalsTransform",
    "StrategyState",
    "Trade",
    # Semantic decorators (new API)
    "alert",
    "apply_fill",
    # Warmup contract
    "assert_warmup_consistency",
    "data_source",
    "data_store",
    "default_registry",
    "detector",
    "entry",
    "executor",
    "exit",
    "feature",
    "fold",
    "get_component",
    "get_component_info",
    "labeler",
    "portfolios_match",
    "register",
    "replay_state",
    "required_warmup_bars",
    "risk",
    # Legacy decorator (deprecated)
    "sf_component",
    "signal_feature",
    "signal_metric",
    "strategy_metric",
    "strategy_store",
    "validator",
    "warmup_bars_of",
]
