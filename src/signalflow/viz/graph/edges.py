"""Edge types for pipeline visualization graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(Enum):
    """Types of edges in a pipeline graph."""

    DATA_FLOW = "data_flow"
    COLUMN_DEPENDENCY = "column"
    SIGNAL_FLOW = "signal"
    FEATURE_CHAIN = "feature"


@dataclass(frozen=True)
class Edge:
    """Edge connecting two nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DATA_FLOW
    columns: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "columns": list(self.columns),
            "metadata": self.metadata,
        }

    def to_cytoscape(self) -> dict[str, Any]:
        """Convert to Cytoscape.js edge format."""
        label = ", ".join(self.columns) if self.columns else ""
        return {
            "data": {
                "id": f"{self.source_id}->{self.target_id}",
                "source": self.source_id,
                "target": self.target_id,
                "label": label,
                "type": self.edge_type.value,
            },
            "classes": self.edge_type.value,
        }
