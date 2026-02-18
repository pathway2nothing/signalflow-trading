"""Configuration loading utilities for SignalFlow.

This module provides unified config loading for all SignalFlow frontends
(sf-kedro, sf-ui, CLI).

Example:
    >>> import signalflow as sf
    >>> config = sf.config.load("grid_sma", conf_path="./conf")
    >>> result = sf.Backtest.from_dict(config).run()

    # Or with typed config:
    >>> flow = sf.config.FlowConfig.from_dict(sf.config.load("grid_sma"))
    >>> flow.detector.type
    'example/sma_cross'

    # Or with DAG config (auto-inferred edges):
    >>> dag = sf.config.FlowDAG.from_dict({
    ...     "nodes": {
    ...         "loader": {"type": "data/loader"},
    ...         "detector": {"type": "signals/detector"},
    ...     }
    ... })
    >>> dag.edges  # auto-inferred: loader → detector
"""

from signalflow.config.dag import (
    Edge,
    EntryMode,
    FlowDAG,
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
    # DAG config
    "Edge",
    "EntryMode",
    "FlowDAG",
    "Node",
    "SignalReconciliation",
    "StrategySubgraph",
    # Chain config (legacy)
    "DataConfig",
    "DetectorConfig",
    "EntryFilterConfig",
    "EntryRuleConfig",
    "ExitRuleConfig",
    "FlowConfig",
    "StrategyConfig",
    # Loader utilities
    "deep_merge",
    "get_flow_info",
    "list_flows",
    "load",
    "load_flow_config",
    "load_yaml",
]
