"""Tests for signalflow.viz.server module."""

import time
import urllib.request

import pytest

from signalflow.viz.graph import DataSourceNode, NodeType, PipelineGraph
from signalflow.viz.server import VizServer


@pytest.fixture
def simple_graph():
    """Simple graph for testing."""
    graph = PipelineGraph()
    graph.add_node(
        DataSourceNode(
            id="data_1",
            name="Test",
            node_type=NodeType.DATA_SOURCE,
        )
    )
    return graph


class TestVizServer:
    def test_server_starts_and_stops(self, simple_graph):
        server = VizServer(simple_graph, port=14141)
        url = server.start(open_browser=False)

        assert url.startswith("http://localhost:")
        assert server._server is not None
        assert server._thread is not None

        server.stop()
        assert server._server is None

    def test_server_serves_html(self, simple_graph):
        server = VizServer(simple_graph, port=14142)
        url = server.start(open_browser=False)

        try:
            # Give server time to start
            time.sleep(0.1)

            # Request the page
            with urllib.request.urlopen(url, timeout=2) as response:
                html = response.read().decode("utf-8")
                assert "<!DOCTYPE html>" in html
                assert "d3.js" in html.lower() or "d3.v7" in html.lower()
        finally:
            server.stop()

    def test_server_health_endpoint(self, simple_graph):
        server = VizServer(simple_graph, port=14143)
        url = server.start(open_browser=False)

        try:
            time.sleep(0.1)
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                data = response.read().decode("utf-8")
                assert "ok" in data
        finally:
            server.stop()

    def test_server_finds_available_port(self, simple_graph):
        # Start first server
        server1 = VizServer(simple_graph, port=14144)
        server1.start(open_browser=False)

        try:
            # Second server should find next available port
            server2 = VizServer(simple_graph, port=14144)
            url2 = server2.start(open_browser=False)

            # Should be on a different port than the requested one
            assert server2.port > 14144
            server2.stop()
        finally:
            server1.stop()
