"""DAG-based flow configuration with automatic input/output inference.

This module provides a DAG (Directed Acyclic Graph) configuration system
where node connections are inferred from inputs/outputs declarations.

Supports complex flows with:
- Multiple loaders (each with its own store)
- Detectors that can receive from features/indicators/other detectors
- Training-only detectors (not passed to strategy)
- Multiple validators per detector OR one validator for multiple detectors
- Strategy with multiple entry/exit rules, model-based decisions, metrics

Example:
    >>> from signalflow.config.dag import FlowDAG, Node
    >>>
    >>> dag = FlowDAG.from_dict({
    ...     "nodes": {
    ...         "binance_loader": {"type": "data/loader", "config": {"exchange": "binance"}},
    ...         "bybit_loader": {"type": "data/loader", "config": {"exchange": "bybit"}},
    ...         "sma_detector": {"type": "signals/detector", "name": "sma_cross"},
    ...         "rsi_detector": {"type": "signals/detector", "name": "rsi", "training_only": True},
    ...         "validator": {"type": "signals/validator"},
    ...         "strategy": {"type": "strategy"},
    ...     }
    ... })
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums for execution modes ────────────────────────────────────────


class EntryMode(str, Enum):
    """How multiple entry rules are combined."""

    SEQUENTIAL = "sequential"  # Check in order, first match wins
    PARALLEL = "parallel"  # Check all, reconcile results
    VOTING = "voting"  # All vote, majority wins


class SignalReconciliation(str, Enum):
    """How multiple signal sources are reconciled in strategy."""

    ANY = "any"  # Any signal triggers entry
    ALL = "all"  # All signals must agree
    WEIGHTED = "weighted"  # Weighted voting
    MODEL = "model"  # Model decides


# ── Default I/O mappings per component type ─────────────────────────


# What each component type produces by default
DEFAULT_OUTPUTS: dict[str, list[str]] = {
    "data/loader": ["ohlcv"],
    "data/store": ["ohlcv"],
    "feature": ["features"],
    "feature/group": ["features"],
    "signals/detector": ["signals"],
    "signals/labeler": ["labels"],
    "signals/validator": ["validated_signals"],
    "strategy/entry": ["entry_decisions"],
    "strategy/exit": ["exit_decisions"],
    "strategy/model": ["model_decisions"],
    "strategy": ["trades", "metrics"],
    "signals/metric": ["signal_metrics"],
    "strategy/metric": ["strategy_metrics"],
}

# What each component type consumes by default
DEFAULT_INPUTS: dict[str, list[str]] = {
    "data/loader": [],  # No inputs, it's a source
    "data/store": [],  # No inputs, it's a source
    "feature": ["ohlcv"],
    "feature/group": ["ohlcv"],
    "signals/detector": ["ohlcv"],  # May also use features, other detectors
    "signals/labeler": ["ohlcv", "signals"],
    "signals/validator": ["signals", "labels"],  # features optional
    "strategy/entry": ["signals"],
    "strategy/exit": ["positions"],  # Internal to strategy
    "strategy/model": ["ohlcv", "signals", "features", "positions"],
    "strategy": ["ohlcv", "signals"],  # Or validated_signals
    "signals/metric": ["signals"],
    "strategy/metric": ["trades"],
}

# Priority order for auto-connecting signal consumers
# When a node needs "signals", prefer validated_signals > signals
SIGNAL_PRIORITY = ["validated_signals", "signals"]

# Node types that should prefer validated_signals over raw signals
# Other types (like labeler) always use raw signals from detector
PREFER_VALIDATED_TYPES = {"strategy", "strategy/entry", "strategy/model"}


@dataclass
class Node:
    """A node in the flow DAG.

    Attributes:
        id: Unique node identifier
        type: Component type (e.g., 'data/loader', 'signals/detector')
        name: Registry name (e.g., 'binance/spot', 'example/sma_cross')
        config: Node-specific configuration
        inputs: Explicit input data types (auto-inferred if not set)
        outputs: Explicit output data types (auto-inferred if not set)
        training_only: If True, this node is only used for training (not execution)
        store: For loaders, the store configuration
        tags: Arbitrary tags for grouping/filtering
    """

    id: str
    type: str
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] | None = None  # None = auto-infer
    outputs: list[str] | None = None  # None = auto-infer
    training_only: bool = False  # Detector only for validator training
    store: dict[str, Any] | None = None  # Store config for loaders
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, node_id: str, data: dict[str, Any]) -> Node:
        """Create Node from dict."""
        return cls(
            id=node_id,
            type=data.get("type", ""),
            name=data.get("name", ""),
            config=data.get("config", {}),
            inputs=data.get("inputs"),
            outputs=data.get("outputs"),
            training_only=data.get("training_only", False),
            store=data.get("store"),
            tags=data.get("tags", []),
        )

    def get_outputs(self) -> list[str]:
        """Get outputs (explicit or default).

        Training-only detectors have special output naming.
        """
        if self.outputs is not None:
            return self.outputs

        default = DEFAULT_OUTPUTS.get(self.type, [])

        # Training-only detectors output "training_signals" instead of "signals"
        if self.training_only and self.type == "signals/detector":
            return ["training_signals"]

        return default

    def get_inputs(self) -> list[str]:
        """Get inputs (explicit or default)."""
        if self.inputs is not None:
            return self.inputs
        return DEFAULT_INPUTS.get(self.type, [])


@dataclass
class Edge:
    """An edge connecting two nodes.

    Attributes:
        source: Source node ID
        target: Target node ID
        data_type: Type of data flowing (e.g., 'ohlcv', 'signals')
        source_output: Specific output port (for multi-output nodes)
        target_input: Specific input port (for multi-input nodes)
    """

    source: str
    target: str
    data_type: str = ""
    source_output: str | None = None
    target_input: str | None = None


@dataclass
class FlowDAG:
    """DAG-based flow configuration.

    Supports automatic edge inference based on node inputs/outputs.

    Attributes:
        id: Flow identifier
        name: Human-readable name
        nodes: Dict of node_id → Node
        edges: List of edges (auto-inferred if not provided)
        config: Global flow config (capital, fee, etc.)
    """

    id: str
    name: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowDAG:
        """Create FlowDAG from dict with auto-inference.

        Args:
            data: Dict with 'nodes' and optional 'edges', 'config'

        Returns:
            FlowDAG with inferred edges if not provided
        """
        flow_id = data.get("id", data.get("flow_id", "flow"))
        name = data.get("name", data.get("flow_name", flow_id))

        # Parse nodes
        nodes_data = data.get("nodes", {})
        nodes: dict[str, Node] = {}

        for node_id, node_data in nodes_data.items():
            nodes[node_id] = Node.from_dict(node_id, node_data)

        # Parse or infer edges
        edges_data = data.get("edges")
        if edges_data is not None:
            # Explicit edges provided
            edges = [
                Edge(
                    source=e.get("source", e.get("from", "")),
                    target=e.get("target", e.get("to", "")),
                    data_type=e.get("data_type", ""),
                    source_output=e.get("source_output"),
                    target_input=e.get("target_input"),
                )
                for e in edges_data
            ]
        else:
            # Auto-infer edges
            edges = cls._infer_edges(nodes)

        return cls(
            id=flow_id,
            name=name,
            nodes=nodes,
            edges=edges,
            config=data.get("config", {}),
        )

    @classmethod
    def _infer_edges(cls, nodes: dict[str, Node]) -> list[Edge]:
        """Infer edges based on node inputs/outputs.

        Algorithm:
        1. Build a map of data_type → producer nodes
        2. For each consumer, find producers for its required inputs
        3. Create edges and warn about auto-connections
        4. Skip training_only nodes when connecting to strategy
        """
        edges: list[Edge] = []
        warnings_issued: list[str] = []

        # Build producer map: data_type → list of (node_id, node)
        producers: dict[str, list[tuple[str, Node]]] = {}
        for node_id, node in nodes.items():
            for output in node.get_outputs():
                producers.setdefault(output, []).append((node_id, node))

        # For each node, find producers for its inputs
        for consumer_id, consumer in nodes.items():
            required_inputs = consumer.get_inputs()

            for required in required_inputs:
                # Special case: signals can come from validated_signals
                # but only for certain consumer types (strategy prefers validated)
                candidates = [required]
                if required == "signals" and consumer.type in PREFER_VALIDATED_TYPES:
                    candidates = SIGNAL_PRIORITY

                producer_found = False
                for candidate in candidates:
                    if candidate in producers:
                        for producer_id, producer in producers[candidate]:
                            if producer_id == consumer_id:
                                continue

                            # Skip training_only detectors for strategy nodes
                            if (
                                producer.training_only
                                and consumer.type in PREFER_VALIDATED_TYPES
                            ):
                                continue

                            edges.append(Edge(
                                source=producer_id,
                                target=consumer_id,
                                data_type=candidate,
                            ))
                            producer_found = True

                            # Warn about auto-connection
                            if consumer.inputs is None:
                                msg = (
                                    f"Auto-connected '{producer_id}' → '{consumer_id}' "
                                    f"(data: {candidate})"
                                )
                                warnings_issued.append(msg)
                        if producer_found:
                            break  # Found producer for this input type

                if not producer_found and required not in ("positions", "features"):
                    # positions is internal to strategy, features often optional
                    msg = f"No producer found for '{consumer_id}' input '{required}'"
                    warnings_issued.append(msg)

        # Issue collected warnings
        for msg in warnings_issued:
            warnings.warn(msg, UserWarning, stacklevel=3)

        return edges

    def get_loaders(self) -> list[Node]:
        """Get all data loader nodes."""
        return [n for n in self.nodes.values() if n.type.startswith("data/")]

    def get_detectors(self, include_training_only: bool = True) -> list[Node]:
        """Get all detector nodes."""
        detectors = [n for n in self.nodes.values() if n.type == "signals/detector"]
        if not include_training_only:
            detectors = [d for d in detectors if not d.training_only]
        return detectors

    def get_validators(self) -> list[Node]:
        """Get all validator nodes."""
        return [n for n in self.nodes.values() if n.type == "signals/validator"]

    def get_strategy_node(self) -> Node | None:
        """Get the strategy node (there should be exactly one)."""
        strategies = [n for n in self.nodes.values() if n.type == "strategy"]
        return strategies[0] if strategies else None

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order (for execution)."""
        # Build adjacency list
        graph: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}

        for edge in self.edges:
            if edge.source in graph and edge.target in graph:
                graph[edge.source].append(edge.target)
                in_degree[edge.target] += 1

        # Kahn's algorithm
        queue = [n for n, d in in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            cycle_nodes = set(self.nodes.keys()) - set(result)
            raise ValueError(f"Cycle detected in DAG: {cycle_nodes}")

        return result

    def get_execution_plan(self) -> list[dict[str, Any]]:
        """Get execution plan with node configs in topological order."""
        order = self.topological_sort()
        plan = []

        for node_id in order:
            node = self.nodes[node_id]
            plan.append({
                "id": node_id,
                "type": node.type,
                "name": node.name,
                "config": node.config,
                "inputs": node.get_inputs(),
                "outputs": node.get_outputs(),
                "training_only": node.training_only,
            })

        return plan

    def validate(self) -> list[str]:
        """Validate DAG structure. Returns list of errors."""
        errors: list[str] = []

        # Check for required node types
        types = {n.type for n in self.nodes.values()}

        if not any(t.startswith("data/") for t in types):
            errors.append("Flow must have at least one data/loader node")

        # Check for cycles (via topological sort)
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        # Check all edge references valid nodes
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge references unknown source: {edge.source}")
            if edge.target not in node_ids:
                errors.append(f"Edge references unknown target: {edge.target}")

        # Strategy validation
        strategy = self.get_strategy_node()
        if strategy:
            subgraph = StrategySubgraph.from_node(strategy)
            errors.extend(subgraph.validate())

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (for serialization)."""
        return {
            "id": self.id,
            "name": self.name,
            "nodes": {
                node_id: {
                    "type": node.type,
                    "name": node.name,
                    "config": node.config,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "training_only": node.training_only,
                    "store": node.store,
                    "tags": node.tags,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "data_type": e.data_type,
                    "source_output": e.source_output,
                    "target_input": e.target_input,
                }
                for e in self.edges
            ],
            "config": self.config,
        }


# ── Strategy Subgraph ────────────────────────────────────────────────


@dataclass
class StrategySubgraph:
    """Internal DAG for strategy execution.

    The strategy node is a composite node with internal structure:

    Signal Sources (from parent DAG):
    - Multiple validated/raw signals from detectors
    - All reconciled before entry decisions

    Entry Layer:
    - entry_rules: Multiple rules, sequential or parallel
    - entry_filters: Applied after entry rules
    - entry_mode: How rules are combined (sequential/parallel/voting)

    Exit Layer:
    - exit_rules: Multiple rules, always parallel
    - Each rule can trigger exit independently

    Model Layer (optional):
    - strategy_model: ML model for entry/exit decisions
    - fallback_entry: Used when model unavailable
    - fallback_exit: Used when model unavailable

    Metrics Layer:
    - metrics: Multiple, parallel, independent
    """

    # Entry configuration
    entry_rules: list[dict[str, Any]] = field(default_factory=list)
    entry_filters: list[dict[str, Any]] = field(default_factory=list)
    entry_mode: EntryMode = EntryMode.SEQUENTIAL

    # Exit configuration
    exit_rules: list[dict[str, Any]] = field(default_factory=list)

    # Model configuration
    strategy_model: dict[str, Any] | None = None
    fallback_entry: dict[str, Any] | None = None
    fallback_exit: dict[str, Any] | None = None

    # Signal reconciliation
    signal_reconciliation: SignalReconciliation = SignalReconciliation.ANY

    # Metrics
    metrics: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_node(cls, node: Node) -> StrategySubgraph:
        """Extract subgraph from strategy node config."""
        config = node.config
        return cls(
            entry_rules=config.get("entry_rules", []),
            entry_filters=config.get("entry_filters", []),
            entry_mode=EntryMode(config.get("entry_mode", "sequential")),
            exit_rules=config.get("exit_rules", []),
            strategy_model=config.get("strategy_model"),
            fallback_entry=config.get("fallback_entry"),
            fallback_exit=config.get("fallback_exit"),
            signal_reconciliation=SignalReconciliation(
                config.get("signal_reconciliation", "any")
            ),
            metrics=config.get("metrics", []),
        )

    def validate(self) -> list[str]:
        """Validate strategy subgraph."""
        errors = []

        # Must have at least one entry rule (or model with fallback)
        if not self.entry_rules and not self.strategy_model:
            errors.append("Strategy must have at least one entry_rule or strategy_model")

        # If model, should have fallbacks
        if self.strategy_model:
            if not self.fallback_entry:
                warnings.warn(
                    "strategy_model without fallback_entry - model failures will block entries",
                    UserWarning,
                )
            if not self.fallback_exit:
                warnings.warn(
                    "strategy_model without fallback_exit - model failures will block exits",
                    UserWarning,
                )

        return errors

    def get_internal_edges(self) -> list[tuple[str, str, str]]:
        """Get internal edges (for visualization).

        Returns list of (source, target, data_type) tuples.
        """
        edges: list[tuple[str, str, str]] = []

        # Signal reconciliation → entry layer
        edges.append(("signal_reconciler", "entry_dispatcher", "reconciled_signals"))

        # Entry layer depends on mode
        if self.entry_mode == EntryMode.SEQUENTIAL:
            # Sequential: chain entry rules
            prev = "entry_dispatcher"
            for i, _ in enumerate(self.entry_rules):
                edges.append((prev, f"entry_rule_{i}", "signals"))
                prev = f"entry_rule_{i}"
            last_entry = prev
        else:
            # Parallel: all entry rules receive signals
            for i, _ in enumerate(self.entry_rules):
                edges.append(("entry_dispatcher", f"entry_rule_{i}", "signals"))
            # Merge results
            for i, _ in enumerate(self.entry_rules):
                edges.append((f"entry_rule_{i}", "entry_merger", "entry_decisions"))
            last_entry = "entry_merger"

        # Entry filters (always sequential)
        if self.entry_filters:
            prev = last_entry
            for j, _ in enumerate(self.entry_filters):
                edges.append((prev, f"entry_filter_{j}", "entry_decisions"))
                prev = f"entry_filter_{j}"
            last_entry = prev

        # Strategy model (if present)
        if self.strategy_model:
            edges.append((last_entry, "strategy_model", "entry_decisions"))
            edges.append(("strategy_model", "position_manager", "model_decisions"))
            # Fallback paths
            if self.fallback_entry:
                edges.append((last_entry, "fallback_entry", "entry_decisions"))
                edges.append(("fallback_entry", "position_manager", "fallback_entry_decisions"))
        else:
            edges.append((last_entry, "position_manager", "entry_decisions"))

        # Exit rules (always parallel)
        for i, _ in enumerate(self.exit_rules):
            edges.append(("position_manager", f"exit_rule_{i}", "positions"))
            edges.append((f"exit_rule_{i}", "exit_merger", "exit_decisions"))

        if self.strategy_model:
            edges.append(("position_manager", "strategy_model", "positions"))
            edges.append(("strategy_model", "exit_merger", "model_exit_decisions"))
            if self.fallback_exit:
                edges.append(("position_manager", "fallback_exit", "positions"))
                edges.append(("fallback_exit", "exit_merger", "fallback_exit_decisions"))

        edges.append(("exit_merger", "runner", "merged_exit_decisions"))

        # Metrics (parallel, independent)
        for i, _ in enumerate(self.metrics):
            edges.append(("runner", f"metric_{i}", "trades"))

        edges.append(("runner", "metrics_output", "trades"))

        return edges
