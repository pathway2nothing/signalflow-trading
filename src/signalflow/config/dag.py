"""Flow configuration with automatic dependency inference.

This module provides a Flow configuration system (DAG-based) where node
connections are inferred from inputs/outputs declarations.

Terminology (Kubeflow-inspired):
- Node: A processing unit (loader, detector, strategy, etc.)
- Dependency: Connection between nodes with artifact type
- Artifact: Data passed between nodes (ohlcv, signals, trades, etc.)
- Flow: The complete DAG of nodes and dependencies
- compile(): Resolve dependencies and validate
- plan(): Get execution order
- run(): Execute the flow

Supports complex flows with:
- Multiple loaders (each with its own store)
- Detectors that can receive from features/indicators/other detectors
- Training-only detectors (not passed to strategy)
- Multiple validators per detector OR one validator for multiple detectors
- Strategy with multiple entry/exit rules, model-based decisions, metrics

Example:
    >>> from signalflow.config import Flow, Node
    >>>
    >>> flow = Flow.from_dict({
    ...     "nodes": {
    ...         "binance_loader": {"type": "data/loader", "config": {"exchange": "binance"}},
    ...         "sma_detector": {"type": "signals/detector", "name": "sma_cross"},
    ...         "strategy": {"type": "strategy"},
    ...     }
    ... })
    >>> flow.compile()  # Resolve dependencies
    >>> flow.plan()     # Get execution order
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from signalflow.backtest import BacktestResult
    from signalflow.config.artifact_cache import ArtifactCache
    from signalflow.core import Signals


# ── Enums for execution modes ────────────────────────────────────────


class EntryMode(StrEnum):
    """How multiple entry rules are combined."""

    SEQUENTIAL = "sequential"  # Check in order, first match wins
    PARALLEL = "parallel"  # Check all, reconcile results
    VOTING = "voting"  # All vote, majority wins


class SignalReconciliation(StrEnum):
    """How multiple signal sources are reconciled in strategy."""

    ANY = "any"  # Any signal triggers entry
    ALL = "all"  # All signals must agree
    WEIGHTED = "weighted"  # Weighted voting
    MODEL = "model"  # Model decides


class FlowMode(StrEnum):
    """Flow execution modes."""

    BACKTEST = "backtest"  # Run backtest simulation
    TRAIN = "train"  # Train ML models (validators, etc.)
    ANALYZE = "analyze"  # Analyze signals without execution


@dataclass
class FlowRunResult:
    """Unified result from Flow.run() across all modes.

    Attributes:
        mode: Execution mode used
        backtest_result: BacktestResult if mode=BACKTEST
        train_result: Training artifacts if mode=TRAIN
        analysis_result: Analysis artifacts if mode=ANALYZE
        artifacts: Dict of intermediate artifacts by node_id.output
        execution_time: Total execution time in seconds
    """

    mode: FlowMode
    backtest_result: BacktestResult | None = None
    train_result: dict[str, Any] | None = None
    analysis_result: dict[str, Any] | None = None
    artifacts: dict[str, pl.DataFrame | Signals] = field(default_factory=dict)
    execution_time: float = 0.0

    @property
    def signals(self) -> dict[str, Any]:
        """Get all signal artifacts."""
        return {k: v for k, v in self.artifacts.items() if k.endswith(".signals") or k.endswith(".validated_signals")}

    @property
    def features(self) -> dict[str, pl.DataFrame]:
        """Get all feature artifacts."""
        return {k: v for k, v in self.artifacts.items() if isinstance(v, pl.DataFrame) and ".features" in k}


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
    "signals/validator": ["signals", "labels", "training_signals"],  # training_signals from training_only detectors
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


# ── Artifact ────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """Data passed between nodes in a Flow.

    Artifacts represent the data flowing through the DAG.
    Each artifact has a name (type) and optional schema/storage hints.

    Attributes:
        name: Artifact type name (e.g., 'ohlcv', 'signals', 'trades')
        dtype: Python/pandas type hint (e.g., 'DataFrame', 'SignalSeries')
        schema: Optional schema for validation
        storage: Storage hint for orchestrators ('memory', 'parquet', 'duckdb')
        producer: Node ID that produces this artifact
    """

    name: str
    dtype: str = "DataFrame"
    schema: dict[str, Any] | None = None
    storage: str = "memory"
    producer: str | None = None

    @classmethod
    def from_output(cls, name: str, producer: str) -> Artifact:
        """Create artifact from node output."""
        # Infer dtype from artifact name
        dtype_map = {
            "ohlcv": "DataFrame",
            "signals": "SignalSeries",
            "validated_signals": "SignalSeries",
            "training_signals": "SignalSeries",
            "labels": "Series",
            "features": "DataFrame",
            "trades": "TradeList",
            "metrics": "dict",
            "positions": "PositionList",
        }
        return cls(
            name=name,
            dtype=dtype_map.get(name, "DataFrame"),
            producer=producer,
        )


# ── Node ────────────────────────────────────────────────────────────


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
class Dependency:
    """A dependency connecting two nodes via an artifact.

    Dependencies represent the flow of data between nodes.
    Each dependency carries an artifact type.

    Attributes:
        source: Source node ID (producer)
        target: Target node ID (consumer)
        artifact: Artifact type being passed (e.g., 'ohlcv', 'signals')
        source_output: Specific output port (for multi-output nodes)
        target_input: Specific input port (for multi-input nodes)
    """

    source: str
    target: str
    artifact: str = ""  # Artifact type (was data_type)
    source_output: str | None = None
    target_input: str | None = None

    @property
    def data_type(self) -> str:
        """Backward compatibility alias for artifact."""
        return self.artifact


# Backward compatibility alias
Edge = Dependency


@dataclass
class Flow:
    """Flow configuration with automatic dependency inference.

    A Flow is a DAG of nodes connected by dependencies (edges).
    Dependencies can be auto-inferred from node inputs/outputs.

    Attributes:
        id: Flow identifier
        name: Human-readable name
        nodes: Dict of node_id → Node
        dependencies: List of dependencies (auto-inferred if not provided)
        config: Global flow config (capital, fee, etc.)
        _compiled: Whether compile() has been called
        _artifacts: Cached artifacts after compile()
    """

    id: str
    name: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)
    dependencies: list[Dependency] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    _compiled: bool = field(default=False, repr=False)
    _artifacts: dict[str, Artifact] = field(default_factory=dict, repr=False)

    @property
    def edges(self) -> list[Dependency]:
        """Backward compatibility alias for dependencies."""
        return self.dependencies

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Flow:
        """Create Flow from dict with auto-inference.

        Args:
            data: Dict with 'nodes' and optional 'dependencies'/'edges', 'config'

        Returns:
            Flow with inferred dependencies if not provided
        """
        flow_id = data.get("id", data.get("flow_id", "flow"))
        name = data.get("name", data.get("flow_name", flow_id))

        # Parse nodes
        nodes_data = data.get("nodes", {})
        nodes: dict[str, Node] = {}

        for node_id, node_data in nodes_data.items():
            nodes[node_id] = Node.from_dict(node_id, node_data)

        # Parse or infer dependencies (support both 'dependencies' and 'edges')
        deps_data = data.get("dependencies") or data.get("edges")
        if deps_data is not None:
            # Explicit dependencies provided
            dependencies = [
                Dependency(
                    source=e.get("source", e.get("from", "")),
                    target=e.get("target", e.get("to", "")),
                    artifact=e.get("artifact", e.get("data_type", "")),
                    source_output=e.get("source_output"),
                    target_input=e.get("target_input"),
                )
                for e in deps_data
            ]
        else:
            # Auto-infer dependencies
            dependencies = cls._infer_dependencies(nodes)

        return cls(
            id=flow_id,
            name=name,
            nodes=nodes,
            dependencies=dependencies,
            config=data.get("config", {}),
        )

    @classmethod
    def _infer_dependencies(cls, nodes: dict[str, Node]) -> list[Dependency]:
        """Infer dependencies based on node inputs/outputs.

        Algorithm:
        1. Build a map of artifact_type → producer nodes
        2. For each consumer, find producers for its required inputs
        3. Create dependencies and warn about auto-connections
        4. Skip training_only nodes when connecting to strategy
        """
        dependencies: list[Dependency] = []
        warnings_issued: list[str] = []

        # Build producer map: artifact_type → list of (node_id, node)
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
                            if producer.training_only and consumer.type in PREFER_VALIDATED_TYPES:
                                continue

                            dependencies.append(
                                Dependency(
                                    source=producer_id,
                                    target=consumer_id,
                                    artifact=candidate,
                                )
                            )
                            producer_found = True

                            # Warn about auto-connection
                            if consumer.inputs is None:
                                msg = f"Auto-connected '{producer_id}' → '{consumer_id}' (data: {candidate})"
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

        return dependencies

    @classmethod
    def _infer_edges(cls, nodes: dict[str, Node]) -> list[Dependency]:
        """Backward compatibility alias for _infer_dependencies."""
        return cls._infer_dependencies(nodes)

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

    def compile(self) -> list[str]:
        """Compile the flow: resolve dependencies and return execution order.

        This validates the DAG structure and returns nodes in topological order.
        After compile(), artifacts are cached for inspection.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If cycle detected in DAG
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}

        for dep in self.dependencies:
            if dep.source in graph and dep.target in graph:
                graph[dep.source].append(dep.target)
                in_degree[dep.target] += 1

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

        # Cache artifacts
        self._artifacts = {}
        for node_id, node in self.nodes.items():
            for output in node.get_outputs():
                self._artifacts[f"{node_id}.{output}"] = Artifact.from_output(output, node_id)

        self._compiled = True
        return result

    def topological_sort(self) -> list[str]:
        """Backward compatibility alias for compile()."""
        return self.compile()

    def plan(self) -> list[dict[str, Any]]:
        """Get execution plan with node configs in topological order.

        Returns:
            List of node execution specs with id, type, config, etc.
        """
        order = self.compile()
        execution_plan = []

        for node_id in order:
            node = self.nodes[node_id]
            execution_plan.append(
                {
                    "id": node_id,
                    "type": node.type,
                    "name": node.name,
                    "config": node.config,
                    "inputs": node.get_inputs(),
                    "outputs": node.get_outputs(),
                    "training_only": node.training_only,
                }
            )

        return execution_plan

    def get_execution_plan(self) -> list[dict[str, Any]]:
        """Backward compatibility alias for plan()."""
        return self.plan()

    @property
    def artifacts(self) -> dict[str, Artifact]:
        """Get all artifacts in the flow.

        Call compile() first to populate artifacts.
        """
        if not self._compiled:
            self.compile()
        return self._artifacts

    def run(
        self,
        *,
        mode: FlowMode = FlowMode.BACKTEST,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
        cancel_event: Event | None = None,
        cache: ArtifactCache | None = None,
        validate_runtime: bool = False,
        **kwargs: Any,
    ) -> FlowRunResult:
        """Execute the flow and return results.

        Executes nodes in topological order, passing artifacts between them.
        Supports multiple modes, progress callbacks, and artifact caching.

        Args:
            mode: Execution mode (BACKTEST, TRAIN, ANALYZE)
            progress_callback: Called with (current_step, total_steps, info)
            cancel_event: Set to gracefully cancel execution
            cache: Optional ArtifactCache for intermediate artifacts
            validate_runtime: Validate artifact schemas at runtime
            **kwargs: Additional arguments for the specific mode

        Returns:
            FlowRunResult with mode-appropriate results and artifacts
        """
        start_time = time.time()

        # Compile to get execution order
        execution_order = self.compile()
        total_steps = len(execution_order)

        # Initialize artifact storage
        artifacts: dict[str, Any] = {}
        result = FlowRunResult(mode=mode)

        # Execute nodes in topological order
        for step, node_id in enumerate(execution_order):
            # Check cancellation
            if cancel_event and cancel_event.is_set():
                break

            # Progress callback
            if progress_callback:
                progress_callback(step + 1, total_steps, {"node": node_id})

            node = self.nodes[node_id]

            # Gather inputs from previous artifacts
            inputs = self._gather_inputs(node, artifacts)

            # Execute node (with cache check)
            outputs = self._execute_node(node, inputs, mode, cache=cache, validate_runtime=validate_runtime)

            # Store outputs
            for output_name in node.get_outputs():
                artifact_key = f"{node_id}.{output_name}"
                if output_name in outputs:
                    artifacts[artifact_key] = outputs[output_name]

        # Build final result based on mode
        result.artifacts = artifacts
        result.execution_time = time.time() - start_time

        if mode == FlowMode.BACKTEST:
            result.backtest_result = self._build_backtest_result(artifacts, kwargs)
        elif mode == FlowMode.TRAIN:
            result.train_result = self._build_train_result(artifacts)
        elif mode == FlowMode.ANALYZE:
            result.analysis_result = self._build_analysis_result(artifacts)

        return result

    def _gather_inputs(
        self,
        node: Node,
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        """Gather input artifacts for a node from previous outputs."""
        inputs: dict[str, Any] = {}

        for dep in self.dependencies:
            if dep.target == node.id:
                artifact_key = f"{dep.source}.{dep.artifact}"
                if artifact_key in artifacts:
                    inputs[dep.artifact] = artifacts[artifact_key]

        return inputs

    def _execute_node(
        self,
        node: Node,
        inputs: dict[str, Any],
        mode: FlowMode,
        cache: ArtifactCache | None = None,
        validate_runtime: bool = False,
    ) -> dict[str, Any]:
        """Execute a single node and return its outputs."""
        # Check cache first
        if cache:
            cached = cache.get(node, inputs)
            if cached is not None:
                return cached

        # Execute based on node type
        outputs: dict[str, Any] = {}

        if node.type == "data/loader":
            outputs = self._execute_loader(node)
        elif node.type == "signals/detector":
            outputs = self._execute_detector(node, inputs, mode)
        elif node.type in ("feature", "feature/group"):
            outputs = self._execute_feature(node, inputs)
        elif node.type == "signals/labeler":
            outputs = self._execute_labeler(node, inputs)
        elif node.type == "signals/validator":
            outputs = self._execute_validator(node, inputs, mode)
        elif node.type == "strategy":
            outputs = self._execute_strategy(node, inputs, mode)

        # Runtime validation
        if validate_runtime:
            self._validate_outputs(node, outputs)

        # Cache outputs
        if cache:
            cache.put(node, inputs, outputs)

        return outputs

    def _execute_loader(self, node: Node) -> dict[str, Any]:
        """Execute a data loader node."""
        from signalflow.api.shortcuts import load as sf_load

        config = node.config
        raw = sf_load(
            source=config.get("source", config.get("store", {}).get("path")),
            pairs=config.get("pairs", ["BTCUSDT"]),
            start=config.get("start"),
            end=config.get("end"),
            timeframe=config.get("timeframe", "1h"),
        )

        # Return the first available DataFrame
        if hasattr(raw, "data") and raw.data:
            for _key, df in raw.data.items():
                if isinstance(df, pl.DataFrame):
                    return {"ohlcv": df}
                if isinstance(df, dict):
                    for sub_df in df.values():
                        return {"ohlcv": sub_df}

        return {"ohlcv": pl.DataFrame()}

    def _execute_detector(
        self,
        node: Node,
        inputs: dict[str, Any],
        mode: FlowMode,
    ) -> dict[str, Any]:
        """Execute a detector node."""
        from signalflow.core import RawData, RawDataView
        from signalflow.core.enums import SfComponentType
        from signalflow.core.registry import default_registry

        # Get detector from registry
        detector = default_registry.create(
            SfComponentType.DETECTOR,
            node.name,
            **node.config,
        )

        # Get OHLCV input
        ohlcv = inputs.get("ohlcv")
        if ohlcv is None:
            raise ValueError(f"Detector {node.id} missing ohlcv input")

        # Wrap in RawData/RawDataView
        raw_data = RawData(
            datetime_start=ohlcv["timestamp"].min(),
            datetime_end=ohlcv["timestamp"].max(),
            pairs=ohlcv["pair"].unique().to_list(),
            data={"spot": ohlcv},
        )

        # Run detection
        signals = detector.run(RawDataView(raw=raw_data))

        # Output name depends on training_only flag
        output_key = "training_signals" if node.training_only else "signals"
        return {output_key: signals}

    def _execute_feature(
        self,
        node: Node,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a feature node."""
        from signalflow.core.enums import SfComponentType
        from signalflow.core.registry import default_registry

        ohlcv = inputs.get("ohlcv")
        if ohlcv is None:
            return {"features": pl.DataFrame()}

        # Get feature from registry
        feature = default_registry.create(
            SfComponentType.FEATURE,
            node.name,
            **node.config,
        )

        # Compute features
        features_df = feature.compute(ohlcv)
        return {"features": features_df}

    def _execute_labeler(
        self,
        node: Node,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a labeler node."""
        from signalflow.core.enums import SfComponentType
        from signalflow.core.registry import default_registry

        signals = inputs.get("signals")
        ohlcv = inputs.get("ohlcv")

        if signals is None or ohlcv is None:
            return {"labels": pl.DataFrame()}

        # Get labeler from registry
        labeler = default_registry.create(
            SfComponentType.LABELER,
            node.name,
            **node.config,
        )

        # Get signals DataFrame
        signals_df = signals.value if hasattr(signals, "value") else signals

        # Run labeling
        labels = labeler.label(signals_df, ohlcv)
        return {"labels": labels}

    def _execute_validator(
        self,
        node: Node,
        inputs: dict[str, Any],
        mode: FlowMode,
    ) -> dict[str, Any]:
        """Execute a validator node."""
        from signalflow.core.enums import SfComponentType
        from signalflow.core.registry import default_registry

        signals = inputs.get("signals")
        labels = inputs.get("labels")

        if signals is None:
            return {"validated_signals": None}

        # Get validator from registry
        validator = default_registry.create(
            SfComponentType.VALIDATOR,
            node.name,
            **node.config,
        )

        # Get signals DataFrame
        signals_df = signals.value if hasattr(signals, "value") else signals

        if mode == FlowMode.TRAIN and labels is not None:
            # Training mode: fit the validator
            validator.fit(signals_df, labels)
            return {"validated_signals": signals, "_validator": validator}

        # Validation mode: predict probabilities
        validated = validator.validate(signals_df)
        return {"validated_signals": validated}

    def _execute_strategy(
        self,
        node: Node,
        inputs: dict[str, Any],
        mode: FlowMode,
    ) -> dict[str, Any]:
        """Execute strategy node (backtest mode only)."""
        if mode != FlowMode.BACKTEST:
            return {"trades": pl.DataFrame(), "metrics": {}}

        # For backtest mode, use the traditional BacktestBuilder path
        from signalflow import Backtest

        config = self._to_backtest_config()

        # Inject pre-computed signals if available
        signals = inputs.get("signals") or inputs.get("validated_signals")
        if signals is not None:
            config["_precomputed_signals"] = signals

        result = Backtest.from_dict(config).run()
        return {
            "trades": result.trades_df if hasattr(result, "trades_df") else pl.DataFrame(),
            "metrics": result.metrics if hasattr(result, "metrics") else {},
        }

    def _validate_outputs(self, node: Node, outputs: dict[str, Any]) -> None:
        """Validate outputs against artifact schemas."""
        from signalflow.config.artifact_schema import get_schema

        for output_name, output_data in outputs.items():
            schema = get_schema(output_name)
            if schema is None:
                continue

            # Get DataFrame from Signals if needed
            df = output_data.value if hasattr(output_data, "value") else output_data
            if not isinstance(df, pl.DataFrame):
                continue

            errors = schema.validate(df)
            if errors:
                raise ValueError(
                    f"Runtime validation failed for {node.id}.{output_name}:\n" + "\n".join(f"  - {e}" for e in errors)
                )

    def _build_backtest_result(
        self,
        artifacts: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> BacktestResult | None:
        """Build BacktestResult from artifacts."""
        # Find trades and metrics from strategy output
        for key, value in artifacts.items():
            if key.endswith(".trades") and isinstance(value, dict):
                # Already have result from _execute_strategy
                pass

        # If no strategy was run, return None
        return None

    def _build_train_result(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        """Build training result from artifacts."""
        return {
            "labels": {k: v for k, v in artifacts.items() if ".labels" in k},
            "features": {k: v for k, v in artifacts.items() if ".features" in k},
            "validators": {k: v for k, v in artifacts.items() if k.endswith("._validator")},
        }

    def _build_analysis_result(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        """Build analysis result from artifacts."""
        return {
            "signals": {k: v for k, v in artifacts.items() if ".signals" in k or ".validated_signals" in k},
            "features": {k: v for k, v in artifacts.items() if ".features" in k},
        }

    def _to_backtest_config(self) -> dict[str, Any]:
        """Convert Flow to BacktestBuilder-compatible config."""
        config: dict[str, Any] = {
            "flow_id": self.id,
            "flow_name": self.name,
            **self.config,
        }

        # Extract node configs
        for _node_id, node in self.nodes.items():
            if node.type == "data/loader":
                config["data"] = {
                    "pairs": node.config.get("pairs", ["BTCUSDT"]),
                    "timeframe": node.config.get("timeframe", "1h"),
                    **node.config,
                }
            elif node.type == "signals/detector":
                config["detector"] = {
                    "type": node.name,
                    **node.config,
                }
            elif node.type == "strategy":
                config["strategy"] = node.config

        return config

    def validate(self) -> list[str]:
        """Validate Flow structure. Returns list of errors."""
        errors: list[str] = []

        # Check for required node types
        types = {n.type for n in self.nodes.values()}

        if not any(t.startswith("data/") for t in types):
            errors.append("Flow must have at least one data/loader node")

        # Check for cycles (via compile)
        try:
            self.compile()
        except ValueError as e:
            errors.append(str(e))

        # Check all dependency references valid nodes
        node_ids = set(self.nodes.keys())
        for dep in self.dependencies:
            if dep.source not in node_ids:
                errors.append(f"Dependency references unknown source: {dep.source}")
            if dep.target not in node_ids:
                errors.append(f"Dependency references unknown target: {dep.target}")

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
            "dependencies": [
                {
                    "source": d.source,
                    "target": d.target,
                    "artifact": d.artifact,
                    "source_output": d.source_output,
                    "target_input": d.target_input,
                }
                for d in self.dependencies
            ],
            # Backward compatibility
            "edges": [
                {
                    "source": d.source,
                    "target": d.target,
                    "data_type": d.artifact,
                    "source_output": d.source_output,
                    "target_input": d.target_input,
                }
                for d in self.dependencies
            ],
            "config": self.config,
        }


# Backward compatibility alias
FlowDAG = Flow


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
            signal_reconciliation=SignalReconciliation(config.get("signal_reconciliation", "any")),
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
                    stacklevel=2,
                )
            if not self.fallback_exit:
                warnings.warn(
                    "strategy_model without fallback_exit - model failures will block exits",
                    UserWarning,
                    stacklevel=2,
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
