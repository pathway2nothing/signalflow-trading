"""Mermaid diagram renderer."""

from __future__ import annotations

from typing import Literal

from signalflow.viz.graph import PipelineGraph


class MermaidRenderer:
    """Render PipelineGraph to Mermaid diagram."""

    def __init__(
        self,
        graph: PipelineGraph,
        direction: Literal["LR", "TB", "RL", "BT"] = "LR",
    ):
        self.graph = graph
        self.direction = direction

    def render(self) -> str:
        """Generate Mermaid diagram code."""
        return self.graph.to_mermaid(direction=self.direction)
