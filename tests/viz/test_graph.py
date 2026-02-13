"""Tests for signalflow.viz.graph module."""

import json

import pytest

from signalflow.viz.graph import (
    DataSourceNode,
    Edge,
    EdgeType,
    FeatureNode,
    NodeType,
    PipelineGraph,
)


class TestNode:
    def test_node_to_dict(self):
        node = DataSourceNode(
            id="data_1",
            name="Test Data",
            node_type=NodeType.DATA_SOURCE,
            exchange="binance",
            data_type="perpetual",
        )
        d = node.to_dict()
        assert d["id"] == "data_1"
        assert d["name"] == "Test Data"
        assert d["type"] == "data_source"
        assert d["exchange"] == "binance"

    def test_node_to_cytoscape(self):
        node = FeatureNode(
            id="feat_1",
            name="RSI 14",
            node_type=NodeType.FEATURE,
            feature_class="RsiFeature",
            requires=("close",),
            outputs=("rsi_14",),
        )
        cy = node.to_cytoscape()
        assert cy["data"]["id"] == "feat_1"
        assert cy["data"]["label"] == "RSI 14"
        assert cy["classes"] == "feature"


class TestEdge:
    def test_edge_to_dict(self):
        edge = Edge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.COLUMN_DEPENDENCY,
            columns=("close", "volume"),
        )
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["type"] == "column"
        assert d["columns"] == ["close", "volume"]

    def test_edge_to_cytoscape(self):
        edge = Edge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.SIGNAL_FLOW,
        )
        cy = edge.to_cytoscape()
        assert cy["data"]["source"] == "a"
        assert cy["data"]["target"] == "b"
        assert cy["classes"] == "signal"


class TestPipelineGraph:
    def test_add_node(self):
        graph = PipelineGraph()
        node = DataSourceNode(
            id="data_1",
            name="Test",
            node_type=NodeType.DATA_SOURCE,
        )
        graph.add_node(node)
        assert "data_1" in graph.nodes
        assert graph.get_node("data_1") == node

    def test_add_edge(self):
        graph = PipelineGraph()
        edge = Edge(source_id="a", target_id="b")
        graph.add_edge(edge)
        assert len(graph.edges) == 1

    def test_get_incoming_outgoing_edges(self):
        graph = PipelineGraph()
        graph.add_edge(Edge(source_id="a", target_id="b"))
        graph.add_edge(Edge(source_id="a", target_id="c"))
        graph.add_edge(Edge(source_id="b", target_id="c"))

        assert len(graph.get_outgoing_edges("a")) == 2
        assert len(graph.get_incoming_edges("c")) == 2
        assert len(graph.get_incoming_edges("a")) == 0

    def test_topological_sort(self):
        graph = PipelineGraph()
        for node_id in ["a", "b", "c"]:
            graph.add_node(
                DataSourceNode(
                    id=node_id,
                    name=node_id,
                    node_type=NodeType.DATA_SOURCE,
                )
            )
        graph.add_edge(Edge(source_id="a", target_id="b"))
        graph.add_edge(Edge(source_id="b", target_id="c"))

        order = graph.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_to_dict(self):
        graph = PipelineGraph()
        graph.add_node(
            DataSourceNode(
                id="data_1",
                name="Test",
                node_type=NodeType.DATA_SOURCE,
            )
        )
        graph.add_edge(Edge(source_id="data_1", target_id="feat_1"))

        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1

    def test_to_cytoscape(self):
        graph = PipelineGraph()
        graph.add_node(
            DataSourceNode(
                id="data_1",
                name="Test",
                node_type=NodeType.DATA_SOURCE,
            )
        )

        cy = graph.to_cytoscape()
        assert "elements" in cy
        # Should be valid JSON
        json.dumps(cy)

    def test_to_mermaid(self):
        graph = PipelineGraph()
        graph.add_node(
            DataSourceNode(
                id="data_1",
                name="Test Data",
                node_type=NodeType.DATA_SOURCE,
            )
        )
        graph.add_node(
            FeatureNode(
                id="feat_1",
                name="RSI",
                node_type=NodeType.FEATURE,
            )
        )
        graph.add_edge(Edge(source_id="data_1", target_id="feat_1", columns=("close",)))

        mermaid = graph.to_mermaid()
        assert "graph LR" in mermaid
        assert "data_1" in mermaid
        assert "feat_1" in mermaid
        assert "close" in mermaid

    def test_merge(self):
        graph1 = PipelineGraph()
        graph1.add_node(
            DataSourceNode(id="a", name="A", node_type=NodeType.DATA_SOURCE)
        )

        graph2 = PipelineGraph()
        graph2.add_node(
            DataSourceNode(id="b", name="B", node_type=NodeType.DATA_SOURCE)
        )
        graph2.add_edge(Edge(source_id="a", target_id="b"))

        merged = graph1.merge(graph2)
        assert "a" in merged.nodes
        assert "b" in merged.nodes
        assert len(merged.edges) == 1
