"""Tests for signalflow.feature.feature_pipeline.FeaturePipeline."""

import polars as pl
import pytest

from signalflow.feature.base import Feature, GlobalFeature
from signalflow.feature.examples import ExampleRsiFeature, ExampleSmaFeature, ExampleGlobalMeanRsiFeature
from signalflow.feature.feature_pipeline import FeaturePipeline


class TestFeaturePipelineCreation:
    def test_empty_features_raises(self):
        with pytest.raises(ValueError, match="at least one feature"):
            FeaturePipeline(features=[])

    def test_single_feature(self, ohlcv_100_bars):
        pipe = FeaturePipeline(features=[ExampleSmaFeature(period=5)])
        result = pipe.compute(ohlcv_100_bars)
        assert "sma_5" in result.columns

    def test_multiple_features(self, ohlcv_100_bars):
        pipe = FeaturePipeline(
            features=[
                ExampleRsiFeature(period=14),
                ExampleSmaFeature(period=5),
            ]
        )
        result = pipe.compute(ohlcv_100_bars)
        assert "rsi_14" in result.columns
        assert "sma_5" in result.columns


class TestFeaturePipelineBatching:
    def test_consecutive_per_pair_batched(self):
        pipe = FeaturePipeline(
            features=[
                ExampleSmaFeature(period=5),
                ExampleSmaFeature(period=10),
            ]
        )
        batches = pipe._group_into_batches()
        assert len(batches) == 1  # single batch

    def test_global_breaks_batch(self):
        pipe = FeaturePipeline(
            features=[
                ExampleSmaFeature(period=5),
                ExampleGlobalMeanRsiFeature(period=14),
                ExampleSmaFeature(period=10),
            ]
        )
        batches = pipe._group_into_batches()
        assert len(batches) == 3  # per-pair, global, per-pair

    def test_all_global_separate(self):
        pipe = FeaturePipeline(
            features=[
                ExampleGlobalMeanRsiFeature(period=14),
                ExampleGlobalMeanRsiFeature(period=21),
            ]
        )
        batches = pipe._group_into_batches()
        assert len(batches) == 2


class TestFeaturePipelineCompute:
    def test_compute_preserves_length(self, ohlcv_100_bars):
        pipe = FeaturePipeline(features=[ExampleSmaFeature(period=5)])
        result = pipe.compute(ohlcv_100_bars)
        assert len(result) == len(ohlcv_100_bars)

    def test_compute_adds_all_output_cols(self, ohlcv_100_bars):
        pipe = FeaturePipeline(features=[ExampleRsiFeature(period=14), ExampleSmaFeature(period=5)])
        result = pipe.compute(ohlcv_100_bars)
        for col in pipe.outputs:
            assert col in result.columns


class TestFeaturePipelineOutputCols:
    def test_outputs_property(self):
        pipe = FeaturePipeline(features=[ExampleRsiFeature(period=14), ExampleSmaFeature(period=5)])
        assert "rsi_14" in pipe.outputs
        assert "sma_5" in pipe.outputs

    def test_output_cols_with_prefix(self):
        pipe = FeaturePipeline(features=[ExampleSmaFeature(period=5)])
        assert pipe.output_cols(prefix="feat_") == ["feat_sma_5"]

    def test_mixed_per_pair_and_global(self, ohlcv_100_bars):
        pipe = FeaturePipeline(
            features=[
                ExampleSmaFeature(period=5),
                ExampleGlobalMeanRsiFeature(period=14),
            ]
        )
        result = pipe.compute(ohlcv_100_bars)
        assert "sma_5" in result.columns
        assert "global_mean_rsi_14" in result.columns
        assert len(result) == len(ohlcv_100_bars)
