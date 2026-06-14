"""Signal detectors."""

from signalflow.detector.base import SignalDetector
from signalflow.detector.classic import SmaCrossDetector
from signalflow.detector.fusion import MarketDropDetector, RevertDetector, ThresholdDetector

__all__ = [
    "SignalDetector",
    "SmaCrossDetector",
    "ThresholdDetector",
    "RevertDetector",
    "MarketDropDetector",
]
