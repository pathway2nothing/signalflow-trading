from signalflow.core import (
    RawData,
    Signals,
    RawDataView,
    Position,
    Trade,
    Portfolio,
    StrategyState,
    Order,
    OrderFill,
    SignalType,
    PositionType,
    SfComponentType,
    DataFrameType,
    RawDataType,
    sf_component,
    get_component,
    default_registry,
    SfTorchModuleMixin,
    SignalsTransform,
)
import signalflow.analytic as analytic
import signalflow.data as data
import signalflow.detector as detector
import signalflow.feature as feature
from signalflow.feature import Feature, FeaturePipeline, GlobalFeature, OffsetFeature
import signalflow.target as target
import signalflow.strategy as strategy
import signalflow.utils as utils
import signalflow.validator as validator


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

    if name == "api":
        import signalflow.api as api
        return api

    if name == "viz":
        import signalflow.viz as viz
        return viz

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
]
