"""Extract full pipeline graph from BacktestBuilder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from signalflow.viz.extractors.base import BaseExtractor
from signalflow.viz.extractors.feature_extractor import FeaturePipelineExtractor
from signalflow.viz.extractors.multi_source_extractor import MultiSourceExtractor
from signalflow.viz.graph import (
    DataSourceNode,
    DetectorNode,
    Edge,
    EdgeType,
    FeatureNode,
    NodeType,
    PipelineGraph,
    RuleNode,
    RunnerNode,
)

if TYPE_CHECKING:
    from signalflow.api.builder import BacktestBuilder


class BacktestExtractor(BaseExtractor):
    """Extract full pipeline graph from BacktestBuilder."""

    def __init__(self, builder: "BacktestBuilder"):
        self.builder = builder

    def extract(self) -> PipelineGraph:
        """Build complete backtest pipeline graph."""
        graph = PipelineGraph(
            metadata={
                "type": "backtest_pipeline",
                "strategy_id": self.builder.strategy_id,
            }
        )

        # 1. Data source nodes
        data_node_ids = self._add_data_nodes(graph)

        # 2. Feature pipeline (if detector has features)
        feature_output_id = None
        if self.builder._detector and hasattr(self.builder._detector, "features"):
            feature_output_id = self._add_feature_nodes(graph, data_node_ids)

        # 3. Detector node
        detector_id = self._add_detector_node(graph, feature_output_id or data_node_ids[0])

        # 4. Entry/Exit rules
        entry_ids, exit_ids = self._add_rule_nodes(graph)

        # 5. Runner node
        self._add_runner_node(graph, detector_id, entry_ids, exit_ids)

        return graph

    def _add_data_nodes(self, graph: PipelineGraph) -> list[str]:
        """Add data source nodes. Returns list of node IDs."""
        node_ids: list[str] = []

        if self.builder._raw is not None:
            # Use multi-source extractor for RawData
            raw_graph = MultiSourceExtractor(self.builder._raw).extract()
            for node in raw_graph.nodes.values():
                graph.add_node(node)
                node_ids.append(node.id)
        elif self.builder._data_params:
            # Lazy data - show configured params
            params = self.builder._data_params
            exchange = params.get("exchange", "exchange")
            data_type = params.get("data_type", "perpetual")
            pairs = params.get("pairs", [])

            node = DataSourceNode(
                id="data_source",
                name=f"{exchange.title() if exchange else 'Data'} ({data_type})",
                node_type=NodeType.DATA_SOURCE,
                exchange=exchange,
                data_type=data_type,
                pairs=tuple(pairs) if pairs else (),
                metadata={
                    "start": str(params.get("start", "")),
                    "end": str(params.get("end", "")),
                    "timeframe": params.get("timeframe", "1h"),
                },
            )
            graph.add_node(node)
            node_ids.append(node.id)
        else:
            # No data configured - add placeholder
            node = DataSourceNode(
                id="data_placeholder",
                name="Data (not configured)",
                node_type=NodeType.DATA_SOURCE,
            )
            graph.add_node(node)
            node_ids.append(node.id)

        return node_ids

    def _add_feature_nodes(self, graph: PipelineGraph, data_node_ids: list[str]) -> str | None:
        """Add feature pipeline nodes. Returns last feature node ID."""
        if not self.builder._detector:
            return None

        features = getattr(self.builder._detector, "features", None)
        if not features:
            return None

        # Check if it's a FeaturePipeline
        from signalflow.feature import FeaturePipeline

        if isinstance(features, FeaturePipeline):
            feature_graph = FeaturePipelineExtractor(features).extract()

            # Re-map raw_data edges to actual data nodes
            for node in feature_graph.nodes.values():
                if node.id != "raw_data":
                    graph.add_node(node)

            last_feature_id = None
            for edge in feature_graph.edges:
                # Replace raw_data with first data node
                source_id = edge.source_id
                if source_id == "raw_data":
                    source_id = data_node_ids[0]

                graph.add_edge(
                    Edge(
                        source_id=source_id,
                        target_id=edge.target_id,
                        edge_type=edge.edge_type,
                        columns=edge.columns,
                    )
                )
                last_feature_id = edge.target_id

            return last_feature_id

        # List of features
        elif isinstance(features, list) and features:
            last_id = None
            for i, feature in enumerate(features):
                node = FeatureNode(
                    id=f"feature_{i}",
                    name=feature.__class__.__name__,
                    node_type=NodeType.FEATURE,
                    feature_class=feature.__class__.__name__,
                    requires=tuple(feature.required_cols()),
                    outputs=tuple(feature.output_cols()),
                )
                graph.add_node(node)

                # Connect to data or previous feature
                source_id = last_id if last_id else data_node_ids[0]
                graph.add_edge(
                    Edge(
                        source_id=source_id,
                        target_id=node.id,
                        edge_type=EdgeType.FEATURE_CHAIN,
                    )
                )
                last_id = node.id

            return last_id

        return None

    def _add_detector_node(self, graph: PipelineGraph, input_id: str) -> str:
        """Add detector node."""
        detector = self.builder._detector
        node_id = "detector"

        if detector:
            name = detector.__class__.__name__
            raw_data_type = getattr(detector, "raw_data_type", "spot")
            if hasattr(raw_data_type, "value"):
                raw_data_type = raw_data_type.value

            node = DetectorNode(
                id=node_id,
                name=name,
                node_type=NodeType.DETECTOR,
                detector_class=name,
                raw_data_type=raw_data_type,
            )
        else:
            node = DetectorNode(
                id=node_id,
                name="Detector (not configured)",
                node_type=NodeType.DETECTOR,
            )

        graph.add_node(node)
        graph.add_edge(
            Edge(
                source_id=input_id,
                target_id=node_id,
                edge_type=EdgeType.DATA_FLOW,
            )
        )

        return node_id

    def _add_rule_nodes(self, graph: PipelineGraph) -> tuple[list[str], list[str]]:
        """Add entry and exit rule nodes."""
        entry_ids: list[str] = []
        exit_ids: list[str] = []

        # Entry rules
        entry_config = self.builder._entry_config
        if entry_config:
            rule_name = entry_config.get("rule", "signal")
            node = RuleNode(
                id="entry_rule",
                name=f"Entry: {rule_name}",
                node_type=NodeType.ENTRY_RULE,
                rule_class=rule_name,
                params={
                    "size_pct": entry_config.get("size_pct"),
                    "max_positions": entry_config.get("max_positions"),
                },
            )
            graph.add_node(node)
            entry_ids.append(node.id)

        # Exit rules
        exit_config = self.builder._exit_config
        if exit_config:
            params: dict[str, Any] = {}
            name_parts = []

            if exit_config.get("tp"):
                params["tp"] = exit_config["tp"]
                name_parts.append(f"TP {exit_config['tp']:.1%}")
            if exit_config.get("sl"):
                params["sl"] = exit_config["sl"]
                name_parts.append(f"SL {exit_config['sl']:.1%}")
            if exit_config.get("trailing"):
                params["trailing"] = exit_config["trailing"]
                name_parts.append(f"Trail {exit_config['trailing']:.1%}")

            name = ", ".join(name_parts) if name_parts else "Exit"

            node = RuleNode(
                id="exit_rule",
                name=f"Exit: {name}",
                node_type=NodeType.EXIT_RULE,
                rule_class="tp_sl",
                params=params,
            )
            graph.add_node(node)
            exit_ids.append(node.id)

        return entry_ids, exit_ids

    def _add_runner_node(
        self,
        graph: PipelineGraph,
        detector_id: str,
        entry_ids: list[str],
        exit_ids: list[str],
    ) -> str:
        """Add runner node."""
        node_id = "runner"
        node = RunnerNode(
            id=node_id,
            name="BacktestRunner",
            node_type=NodeType.RUNNER,
            runner_class="BacktestRunner",
            entry_rules=tuple(entry_ids),
            exit_rules=tuple(exit_ids),
            metadata={
                "capital": self.builder._capital,
                "fee": self.builder._fee,
            },
        )
        graph.add_node(node)

        # Connect detector → runner (signals flow)
        graph.add_edge(
            Edge(
                source_id=detector_id,
                target_id=node_id,
                edge_type=EdgeType.SIGNAL_FLOW,
            )
        )

        # Connect rules → runner
        for rule_id in entry_ids + exit_ids:
            graph.add_edge(
                Edge(
                    source_id=rule_id,
                    target_id=node_id,
                    edge_type=EdgeType.DATA_FLOW,
                )
            )

        return node_id
