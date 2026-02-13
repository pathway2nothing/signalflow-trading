"""Base extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from signalflow.viz.graph import PipelineGraph


class BaseExtractor(ABC):
    """Abstract base class for graph extractors."""

    @abstractmethod
    def extract(self) -> PipelineGraph:
        """Extract graph from source object."""
        ...
