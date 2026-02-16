import signalflow.analytic as analytic
import signalflow.data as data
import signalflow.detector as detector
import signalflow.feature as feature
import signalflow.strategy as strategy
import signalflow.target as target
import signalflow.utils as utils
import signalflow.validator as validator
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
    sf_component,
)
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

    if name == "viz":
        import signalflow.viz as viz

        return viz

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

    raise AttributeError(f"module 'signalflow' has no attribute {name!r}")


__all__ = [
    # Submodules
    "analytic",
    "api",
    "core",
    "data",
    "detector",
    "feature",
    "strategy",
    "target",
    "utils",
    "validator",
    "viz",
    # Core containers
    "RawData",
    "Signals",
    "RawDataView",
    "Position",
    "Trade",
    "Portfolio",
    "StrategyState",
    "Order",
    "OrderFill",
    # Enums
    "SignalType",
    "PositionType",
    "SfComponentType",
    "DataFrameType",
    "RawDataType",
    # Registry
    "sf_component",
    "get_component",
    "default_registry",
    # Features
    "Feature",
    "FeaturePipeline",
    "GlobalFeature",
    "OffsetFeature",
    # Torch
    "SfTorchModuleMixin",
    "SignalsTransform",
    # High-level API (lazy loaded)
    "Backtest",
    "BacktestBuilder",
    "BacktestResult",
    "backtest",
    "load",
    "load_artifact",
    # Flow API (unified pipeline)
    "flow",
    "FlowBuilder",
    "FlowResult",
    # Metric nodes
    "FeatureMetrics",
    "SignalMetrics",
    "LabelMetrics",
    "ValidationMetrics",
    "BacktestMetrics",
    "LiveMetrics",
]
