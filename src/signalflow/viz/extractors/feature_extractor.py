"""Extract graph from FeaturePipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from signalflow.core.registry import default_registry
from signalflow.viz.extractors.base import BaseExtractor
from signalflow.viz.graph import (
    DataSourceNode,
    Edge,
    EdgeType,
    FeatureNode,
    NodeType,
    PipelineGraph,
)

if TYPE_CHECKING:
    from signalflow.feature import FeaturePipeline
    from signalflow.feature.base import Feature


class FeaturePipelineExtractor(BaseExtractor):
    """Extract graph from FeaturePipeline."""

    def __init__(self, pipeline: FeaturePipeline):
        self.pipeline = pipeline

    def extract(self) -> PipelineGraph:
        """Build graph from FeaturePipeline features."""
        graph = PipelineGraph(metadata={"type": "feature_pipeline"})

        # Get raw data columns as input
        raw_cols = default_registry.get_raw_data_columns(self.pipeline.raw_data_type)
        data_type = getattr(self.pipeline.raw_data_type, "value", str(self.pipeline.raw_data_type))

        # Add input node
        input_node = DataSourceNode(
            id="raw_data",
            name=f"Raw Data ({data_type})",
            node_type=NodeType.DATA_SOURCE,
            data_type=data_type,
            columns=frozenset(raw_cols),
        )
        graph.add_node(input_node)

        # Track column sources: column_name -> node_id
        column_sources: dict[str, str] = {col: "raw_data" for col in raw_cols}

        # Process each feature
        for i, feature in enumerate(self.pipeline.features):
            node = self._feature_to_node(feature, i)
            graph.add_node(node)

            # Add edges from required columns
            for req_col in feature.required_cols():
                source_id = column_sources.get(req_col, "raw_data")
                graph.add_edge(
                    Edge(
                        source_id=source_id,
                        target_id=node.id,
                        edge_type=EdgeType.COLUMN_DEPENDENCY,
                        columns=(req_col,),
                    )
                )

            # Register output columns
            for out_col in feature.output_cols():
                column_sources[out_col] = node.id

        return graph

    def _feature_to_node(self, feature: Feature, index: int) -> FeatureNode:
        """Convert Feature instance to FeatureNode."""
        from signalflow.feature.base import GlobalFeature

        cls_name = feature.__class__.__name__
        # Build unique ID
        node_id = f"feature_{index}_{cls_name}"

        # Extract params (exclude internal fields)
        exclude_fields = {"group_col", "ts_col", "normalized", "norm_period", "sources"}
        params = {k: v for k, v in feature.__dict__.items() if not k.startswith("_") and k not in exclude_fields}

        # Build display name
        name_parts = [cls_name]
        if hasattr(feature, "period"):
            name_parts.append(str(feature.period))
        display_name = " ".join(name_parts)

        return FeatureNode(
            id=node_id,
            name=display_name,
            node_type=NodeType.FEATURE,
            feature_class=cls_name,
            requires=tuple(feature.required_cols()),
            outputs=tuple(feature.output_cols()),
            params=params,
            is_global=isinstance(feature, GlobalFeature),
        )
