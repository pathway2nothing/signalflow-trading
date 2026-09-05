"""Data layer: Dataset container and source plugins."""

from signalflow.data.dataset import Bar, Dataset, data
from signalflow.data.source import BinanceSource, CachedSource, Source, SyntheticSource
from signalflow.data.source import MemorySource as MemorySource  # deprecated alias

__all__ = ["Bar", "BinanceSource", "CachedSource", "Dataset", "Source", "SyntheticSource", "data"]
