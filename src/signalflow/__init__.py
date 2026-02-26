# Allow sub-packages from other directories (sf-ta, sf-nn) to be found
# under the signalflow namespace when they are on sys.path.
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from signalflow.core import (
    DataFrameType,
    Order,
    OrderFill,
    Portfolio,
    Position,
    PositionType,
    RawData,
    RawDataType,
    RawDataView,
    SfComponentType,
    SfTorchModuleMixin,
    Signals,
    SignalsTransform,
    SignalType,
    StrategyState,
    Trade,
    default_registry,
    get_component,
    # Semantic decorators (new API) - use as @sf.detector, @sf.entry, etc.
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
import signalflow.analytic as analytic
import signalflow.data as data
import signalflow.detector as detectors
import signalflow.feature as features
import signalflow.strategy as strategy
import signalflow.target as target
import signalflow.utils as utils
import signalflow.validator as validators
from signalflow.feature import Feature, FeaturePipeline, GlobalFeature, OffsetFeature

# =============================================================================
# Lazy imports for high-level API
# =============================================================================
# These are loaded on first access to avoid import overhead.
# IMPORTANT: Only api/ module is lazy-loaded. Other modules (detector, feature,
# etc.) must stay eager for autodiscover() to work correctly.


def __getattr__(name: str):
    """Lazy load API module components."""
    if name == "Backtest":
        from signalflow.api.builder import Backtest

        return Backtest

    if name == "BacktestBuilder":
        from signalflow.api.builder import BacktestBuilder

        return BacktestBuilder

    if name == "BacktestResult":
        from signalflow.api.result import BacktestResult

        return BacktestResult

    if name == "backtest":
        from signalflow.api.shortcuts import backtest

        return backtest

    if name == "load":
        from signalflow.api.shortcuts import load

        return load

    if name == "load_artifact":
        from signalflow.api.shortcuts import load_artifact

        return load_artifact

    if name == "api":
        import signalflow.api as api

        return api

    # Config module
    if name == "config":
        import signalflow.config as config

        return config

    # Flow API (unified pipeline)
    if name == "flow":
        from signalflow.api.flow import flow

        return flow

    if name == "FlowBuilder":
        from signalflow.api.flow import FlowBuilder

        return FlowBuilder

    if name == "FlowResult":
        from signalflow.api.flow import FlowResult

        return FlowResult

    # Metric nodes
    if name == "FeatureMetrics":
        from signalflow.api.flow import FeatureMetrics

        return FeatureMetrics

    if name == "SignalMetrics":
        from signalflow.api.flow import SignalMetrics

        return SignalMetrics

    if name == "LabelMetrics":
        from signalflow.api.flow import LabelMetrics

        return LabelMetrics

    if name == "ValidationMetrics":
        from signalflow.api.flow import ValidationMetrics

        return ValidationMetrics

    if name == "BacktestMetrics":
        from signalflow.api.flow import BacktestMetrics

        return BacktestMetrics

    if name == "LiveMetrics":
        from signalflow.api.flow import LiveMetrics

        return LiveMetrics

    # Risk management
    if name == "RiskManager":
        from signalflow.strategy.risk.manager import RiskManager

        return RiskManager

    if name == "RiskLimit":
        from signalflow.strategy.risk.limits import RiskLimit

        return RiskLimit

    if name == "MaxLeverageLimit":
        from signalflow.strategy.risk.limits import MaxLeverageLimit

        return MaxLeverageLimit

    if name == "MaxPositionsLimit":
        from signalflow.strategy.risk.limits import MaxPositionsLimit

        return MaxPositionsLimit

    if name == "PairExposureLimit":
        from signalflow.strategy.risk.limits import PairExposureLimit

        return PairExposureLimit

    if name == "DailyLossLimit":
        from signalflow.strategy.risk.limits import DailyLossLimit

        return DailyLossLimit

    raise AttributeError(f"module 'signalflow' has no attribute {name!r}")


__all__ = [
    # High-level API
    "Backtest",
    "BacktestBuilder",
    "BacktestMetrics",
    "BacktestResult",
    "FlowBuilder",
    "FlowResult",
    "backtest",
    "flow",
    "load",
    "load_artifact",
    # Metrics nodes
    "FeatureMetrics",
    "LabelMetrics",
    "LiveMetrics",
    "SignalMetrics",
    "ValidationMetrics",
    # Containers
    "Order",
    "OrderFill",
    "Portfolio",
    "Position",
    "RawData",
    "RawDataType",
    "RawDataView",
    "Signals",
    "StrategyState",
    "Trade",
    # Enums
    "DataFrameType",
    "PositionType",
    "SfComponentType",
    "SignalType",
    # Features
    "Feature",
    "FeaturePipeline",
    "GlobalFeature",
    "OffsetFeature",
    # Risk
    "DailyLossLimit",
    "MaxLeverageLimit",
    "MaxPositionsLimit",
    "PairExposureLimit",
    "RiskLimit",
    "RiskManager",
    # Registry
    "default_registry",
    "get_component",
    # Semantic decorators (new API) - @sf.detector, @sf.entry, etc.
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
    # Sub-packages
    "analytic",
    "api",
    "config",
    "core",
    "data",
    "detectors",
    "features",
    "strategy",
    "target",
    "utils",
    "validators",
    # Other
    "SfTorchModuleMixin",
    "SignalsTransform",
]
