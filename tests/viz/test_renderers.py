"""Tests for signalflow.viz.renderers module."""

import json
import tempfile
from pathlib import Path

import pytest

from signalflow.viz.graph import (
    DataSourceNode,
    Edge,
    EdgeType,
    FeatureNode,
    NodeType,
    PipelineGraph,
)
from signalflow.viz.renderers import HtmlRenderer, MermaidRenderer


@pytest.fixture
def sample_graph():
    """Sample graph for testing renderers."""
    graph = PipelineGraph(metadata={"type": "test"})
    graph.add_node(
        DataSourceNode(
            id="data_1",
            name="Binance Perpetual",
            node_type=NodeType.DATA_SOURCE,
            exchange="binance",
            data_type="perpetual",
        )
    )
    graph.add_node(
        FeatureNode(
            id="feat_rsi",
            name="RSI 14",
            node_type=NodeType.FEATURE,
            feature_class="RsiFeature",
            requires=("close",),
            outputs=("rsi_14",),
        )
    )
    graph.add_edge(
        Edge(
            source_id="data_1",
            target_id="feat_rsi",
            edge_type=EdgeType.COLUMN_DEPENDENCY,
            columns=("close",),
        )
    )
    return graph


class TestMermaidRenderer:
    def test_render_basic(self, sample_graph):
        renderer = MermaidRenderer(sample_graph)
        mermaid = renderer.render()

        assert "graph LR" in mermaid
        assert "data_1" in mermaid
        assert "feat_rsi" in mermaid

    def test_render_direction(self, sample_graph):
        renderer = MermaidRenderer(sample_graph, direction="TB")
        mermaid = renderer.render()
        assert "graph TB" in mermaid


class TestHtmlRenderer:
    def test_render_returns_html(self, sample_graph):
        renderer = HtmlRenderer(sample_graph)
        html = renderer.render()

        assert "<!DOCTYPE html>" in html
        assert "d3.js" in html.lower() or "d3.v7" in html.lower()
        assert "SignalFlow" in html

    def test_render_includes_graph_json(self, sample_graph):
        renderer = HtmlRenderer(sample_graph)
        html = renderer.render()

        # Should contain JSON data
        assert "elements" in html
        assert "data_1" in html
        assert "feat_rsi" in html

    def test_render_to_file(self, sample_graph, tmp_path):
        output_path = tmp_path / "pipeline.html"
        renderer = HtmlRenderer(sample_graph)
        html = renderer.render(output_path=output_path)

        assert output_path.exists()
        assert output_path.read_text() == html

    def test_render_has_sidebar_and_controls(self, sample_graph):
        renderer = HtmlRenderer(sample_graph)
        html = renderer.render()

        # Should have sidebar with stats
        assert "node-count" in html
        assert "edge-count" in html
        # Should have zoom controls
        assert "btn-fit" in html
        assert "btn-zoom-in" in html
