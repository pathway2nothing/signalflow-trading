"""Configuration loading utilities for SignalFlow.

This module provides unified config loading for all SignalFlow frontends
(sf-kedro, sf-ui, CLI).

Terminology (Kubeflow-inspired):
- Flow: The complete DAG configuration
- Node: A processing unit (loader, detector, strategy, etc.)
- Dependency: Connection between nodes with artifact type
- Artifact: Data passed between nodes (ohlcv, signals, trades, etc.)

Example:
    >>> import signalflow as sf
    >>> config = sf.config.load("grid_sma", conf_path="./conf")
    >>> result = sf.Backtest.from_dict(config).run()

    # Or with Flow (DAG-style, auto-inferred dependencies):
    >>> flow = sf.config.Flow.from_dict({
    ...     "nodes": {
    ...         "loader": {"type": "data/loader"},
    ...         "detector": {"type": "signals/detector"},
    ...     }
    ... })
    >>> flow.compile()  # Resolve dependencies
    >>> flow.plan()     # Get execution order
    >>> flow.run()      # Execute backtest
"""

from signalflow.config.artifact_cache import ArtifactCache
from signalflow.config.artifact_schema import (
    FEATURES_SCHEMA,
    LABELS_SCHEMA,
    OHLCV_SCHEMA,
    SIGNALS_SCHEMA,
    TRADES_SCHEMA,
    ArtifactSchema,
    ArtifactType,
    ColumnSchema,
    get_schema,
)
from signalflow.config.dag import (
    # New names (Kubeflow-inspired)
    Artifact,
    Dependency,
    # Backward compatibility aliases
    Edge,  # alias for Dependency
    # Other exports
    EntryMode,
    Flow,
    FlowDAG,  # alias for Flow
    FlowMode,
    FlowRunResult,
    Node,
    SignalReconciliation,
    StrategySubgraph,
)
from signalflow.config.flow import (
    DataConfig,
    DetectorConfig,
    EntryFilterConfig,
    EntryRuleConfig,
    ExitRuleConfig,
    FlowConfig,
    StrategyConfig,
)
from signalflow.config.loader import (
    deep_merge,
    get_flow_info,
    list_flows,
    load_flow_config,
    load_yaml,
)

# Convenience alias
load = load_flow_config

__all__ = [
    "FEATURES_SCHEMA",
    "LABELS_SCHEMA",
    "OHLCV_SCHEMA",
    "SIGNALS_SCHEMA",
    "TRADES_SCHEMA",
    "Artifact",
    "ArtifactCache",
    "ArtifactSchema",
    "ArtifactType",
    "ColumnSchema",
    "DataConfig",
    "Dependency",
    "DetectorConfig",
    "Edge",
    "EntryFilterConfig",
    "EntryMode",
    "EntryRuleConfig",
    "ExitRuleConfig",
    "Flow",
    "FlowConfig",
    "FlowDAG",
    "FlowMode",
    "FlowRunResult",
    "Node",
    "SignalReconciliation",
    "StrategyConfig",
    "StrategySubgraph",
    "deep_merge",
    "get_flow_info",
    "get_schema",
    "list_flows",
    "load",
    "load_flow_config",
    "load_yaml",
]
