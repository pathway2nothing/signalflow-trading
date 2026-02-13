"""Pipeline graph container for visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from signalflow.viz.graph.edges import Edge, EdgeType
from signalflow.viz.graph.nodes import Node, NodeType


@dataclass
class PipelineGraph:
    """DAG representation of a SignalFlow pipeline."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Node | None:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def topological_sort(self) -> list[str]:
        """Return node IDs in topological order."""
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            for edge in self.get_outgoing_edges(node_id):
                if edge.target_id in in_degree:
                    in_degree[edge.target_id] -= 1
                    if in_degree[edge.target_id] == 0:
                        queue.append(edge.target_id)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Export graph as JSON-serializable dict."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    def to_cytoscape(self) -> dict[str, Any]:
        """Export in Cytoscape.js format."""
        elements = []
        for node in self.nodes.values():
            elements.append(node.to_cytoscape())
        for edge in self.edges:
            elements.append(edge.to_cytoscape())
        return {"elements": elements}

    def to_mermaid(self, direction: str = "LR") -> str:
        """Export as Mermaid diagram."""
        lines = [f"graph {direction}"]

        # Node shape mapping
        shape_map = {
            NodeType.DATA_SOURCE: ("([", "])"),
            NodeType.FEATURE: ("[", "]"),
            NodeType.DETECTOR: ("{{", "}}"),
            NodeType.LABELER: ("[/", "/]"),
            NodeType.VALIDATOR: (">", "]"),
            NodeType.RUNNER: ("[[", "]]"),
            NodeType.ENTRY_RULE: ("(", ")"),
            NodeType.EXIT_RULE: ("(", ")"),
        }

        # Group nodes by type for subgraphs
        type_groups: dict[NodeType, list[Node]] = {}
        for node in self.nodes.values():
            if node.node_type not in type_groups:
                type_groups[node.node_type] = []
            type_groups[node.node_type].append(node)

        # Define subgraph labels
        subgraph_names = {
            NodeType.DATA_SOURCE: "Data Sources",
            NodeType.FEATURE: "Features",
            NodeType.DETECTOR: "Detector",
            NodeType.LABELER: "Labeler",
            NodeType.VALIDATOR: "Validator",
            NodeType.RUNNER: "Runner",
            NodeType.ENTRY_RULE: "Entry Rules",
            NodeType.EXIT_RULE: "Exit Rules",
        }

        # Render subgraphs
        for node_type in NodeType:
            nodes = type_groups.get(node_type, [])
            if not nodes:
                continue

            subgraph_name = subgraph_names.get(node_type, node_type.value)
            lines.append(f"    subgraph {subgraph_name}")

            left, right = shape_map.get(node_type, ("[", "]"))
            for node in nodes:
                # Escape special chars in name
                safe_name = node.name.replace('"', "'")
                lines.append(f'        {node.id}{left}"{safe_name}"{right}')

            lines.append("    end")

        # Add edges
        for edge in self.edges:
            if edge.columns:
                label = ", ".join(edge.columns[:3])
                if len(edge.columns) > 3:
                    label += f" +{len(edge.columns) - 3}"
                lines.append(f"    {edge.source_id} -->|{label}| {edge.target_id}")
            else:
                lines.append(f"    {edge.source_id} --> {edge.target_id}")

        return "\n".join(lines)

    def merge(self, other: PipelineGraph) -> PipelineGraph:
        """Merge another graph into this one."""
        result = PipelineGraph(
            nodes={**self.nodes, **other.nodes},
            edges=self.edges + other.edges,
            metadata={**self.metadata, **other.metadata},
        )
        return result
