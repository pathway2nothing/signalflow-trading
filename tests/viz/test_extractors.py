"""Tests for signalflow.viz.extractors module."""


from signalflow.viz.extractors import (
    BacktestExtractor,
    FeaturePipelineExtractor,
    MultiSourceExtractor,
)
from signalflow.viz.graph import NodeType


class TestFeaturePipelineExtractor:
    def test_extracts_input_node(self, sample_feature_pipeline):
        graph = FeaturePipelineExtractor(sample_feature_pipeline).extract()

        # Should have raw_data input node
        assert "raw_data" in graph.nodes
        assert graph.nodes["raw_data"].node_type == NodeType.DATA_SOURCE

    def test_extracts_feature_nodes(self, sample_feature_pipeline):
        graph = FeaturePipelineExtractor(sample_feature_pipeline).extract()

        # Should have 2 feature nodes
        feature_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.FEATURE]
        assert len(feature_nodes) == 2

    def test_feature_nodes_have_requires_outputs(self, sample_feature_pipeline):
        graph = FeaturePipelineExtractor(sample_feature_pipeline).extract()

        feature_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.FEATURE]
        for node in feature_nodes:
            assert len(node.requires) > 0
            assert len(node.outputs) > 0

    def test_creates_column_dependency_edges(self, sample_feature_pipeline):
        graph = FeaturePipelineExtractor(sample_feature_pipeline).extract()

        # Should have edges from raw_data to features
        assert len(graph.edges) >= 2

    def test_graph_metadata(self, sample_feature_pipeline):
        graph = FeaturePipelineExtractor(sample_feature_pipeline).extract()
        assert graph.metadata["type"] == "feature_pipeline"


class TestMultiSourceExtractor:
    def test_extracts_nested_sources(self, multi_source_raw_data):
        graph = MultiSourceExtractor(multi_source_raw_data).extract()

        # Should have 2 data source nodes (binance, okx)
        data_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.DATA_SOURCE]
        assert len(data_nodes) == 2

        # Check exchange attribute
        exchanges = {n.exchange for n in data_nodes}
        assert "binance" in exchanges
        assert "okx" in exchanges

    def test_extracts_flat_source(self, flat_raw_data):
        graph = MultiSourceExtractor(flat_raw_data).extract()

        # Should have 1 data source node
        data_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.DATA_SOURCE]
        assert len(data_nodes) == 1
        assert data_nodes[0].exchange is None

    def test_metadata_includes_pairs(self, multi_source_raw_data):
        graph = MultiSourceExtractor(multi_source_raw_data).extract()
        assert "pairs" in graph.metadata
        assert "BTCUSDT" in graph.metadata["pairs"]


class TestBacktestExtractor:
    def test_extracts_from_builder_with_data_params(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")

        graph = BacktestExtractor(builder).extract()

        # Should have data source node
        data_nodes = graph.get_nodes_by_type(NodeType.DATA_SOURCE)
        assert len(data_nodes) == 1
        assert data_nodes[0].exchange == "binance"

    def test_extracts_detector_node(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        # Note: detector not configured

        graph = BacktestExtractor(builder).extract()

        # Should have detector node (even if not configured)
        detector_nodes = graph.get_nodes_by_type(NodeType.DETECTOR)
        assert len(detector_nodes) == 1

    def test_extracts_runner_node(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")

        graph = BacktestExtractor(builder).extract()

        runner_nodes = graph.get_nodes_by_type(NodeType.RUNNER)
        assert len(runner_nodes) == 1
        assert runner_nodes[0].runner_class == "BacktestRunner"

    def test_extracts_entry_exit_rules(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        builder.entry(size_pct=0.1)
        builder.exit(tp=0.03, sl=0.015)

        graph = BacktestExtractor(builder).extract()

        entry_nodes = graph.get_nodes_by_type(NodeType.ENTRY_RULE)
        exit_nodes = graph.get_nodes_by_type(NodeType.EXIT_RULE)
        assert len(entry_nodes) == 1
        assert len(exit_nodes) == 1

    def test_graph_has_metadata(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="my_strategy")

        graph = BacktestExtractor(builder).extract()
        assert graph.metadata["strategy_id"] == "my_strategy"
        assert graph.metadata["type"] == "backtest_pipeline"
