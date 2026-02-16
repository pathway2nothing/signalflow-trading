"""Extract data flow graph from multi-source RawData."""

from __future__ import annotations

from typing import TYPE_CHECKING

from signalflow.viz.extractors.base import BaseExtractor
from signalflow.viz.graph import (
    DataSourceNode,
    NodeType,
    PipelineGraph,
)

if TYPE_CHECKING:
    from signalflow.core import RawData


class MultiSourceExtractor(BaseExtractor):
    """Extract data flow graph from multi-source RawData."""

    def __init__(self, raw: RawData):
        self.raw = raw

    def extract(self) -> PipelineGraph:
        """Build graph showing data sources and their structure."""
        graph = PipelineGraph(
            metadata={
                "type": "multi_source",
                "datetime_start": str(self.raw.datetime_start),
                "datetime_end": str(self.raw.datetime_end),
                "pairs": list(self.raw.pairs),
            }
        )

        for data_type in self.raw.keys():
            sources = self.raw.sources(data_type)

            if len(sources) == 1 and sources[0] == "default":
                # Single source (flat structure)
                df = self.raw.get(data_type)
                node = DataSourceNode(
                    id=f"data_{data_type}",
                    name=data_type.title(),
                    node_type=NodeType.DATA_SOURCE,
                    data_type=data_type,
                    columns=frozenset(df.columns),
                    pairs=tuple(self.raw.pairs),
                    metadata={
                        "rows": df.height,
                        "is_nested": False,
                    },
                )
                graph.add_node(node)
            else:
                # Multi-source (nested structure)
                for source in sources:
                    df = self.raw.get(data_type, source=source)
                    node = DataSourceNode(
                        id=f"data_{data_type}_{source}",
                        name=f"{source.title()} {data_type.title()}",
                        node_type=NodeType.DATA_SOURCE,
                        exchange=source,
                        data_type=data_type,
                        columns=frozenset(df.columns),
                        pairs=tuple(self.raw.pairs),
                        metadata={
                            "rows": df.height,
                            "is_nested": True,
                        },
                    )
                    graph.add_node(node)

        return graph
