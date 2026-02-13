"""Tests for signalflow.viz public API."""

import pytest

from signalflow import viz
from signalflow.viz.graph import PipelineGraph


class TestPipelineFunction:
    def test_pipeline_with_feature_pipeline(self, sample_feature_pipeline):
        result = viz.pipeline(sample_feature_pipeline, format="mermaid", show=False)
        assert "graph LR" in result
        assert "RsiFeature" in result or "rsi" in result.lower()

    def test_pipeline_with_backtest_builder(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        builder.entry(size_pct=0.1)
        builder.exit(tp=0.03, sl=0.015)

        result = viz.pipeline(builder, format="mermaid", show=False)
        assert "graph LR" in result
        assert "Runner" in result or "runner" in result

    def test_pipeline_json_format(self, sample_feature_pipeline):
        result = viz.pipeline(sample_feature_pipeline, format="json", show=False)
        assert isinstance(result, PipelineGraph)
        assert len(result.nodes) > 0

    def test_pipeline_unsupported_type(self):
        with pytest.raises(TypeError):
            viz.pipeline("not a pipeline", show=False)


class TestFeaturesFunction:
    def test_features_returns_mermaid(self, sample_feature_pipeline):
        result = viz.features(sample_feature_pipeline, format="mermaid", show=False)
        assert "graph LR" in result

    def test_features_html_format(self, sample_feature_pipeline, tmp_path):
        output_path = tmp_path / "features.html"
        result = viz.features(
            sample_feature_pipeline,
            format="html",
            output=str(output_path),
            show=False,
        )
        assert output_path.exists()
        assert "<!DOCTYPE html>" in result


class TestDataFlowFunction:
    def test_data_flow_multi_source(self, multi_source_raw_data):
        result = viz.data_flow(multi_source_raw_data, format="mermaid", show=False)
        assert "graph LR" in result
        assert "binance" in result.lower() or "Binance" in result

    def test_data_flow_flat_source(self, flat_raw_data):
        result = viz.data_flow(flat_raw_data, format="mermaid", show=False)
        assert "graph LR" in result


class TestBuilderVisualize:
    def test_visualize_method(self):
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id="test")
        builder.data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")

        result = builder.visualize(format="mermaid", show=False)
        assert "graph LR" in result
