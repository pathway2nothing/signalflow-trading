# Preset detectors (backward compat)
from signalflow.detector.anomaly_detector import AnomalyDetector
from signalflow.detector.base import SignalDetector

# Funding rate detector
from signalflow.detector.funding_rate import FundingRateDetector
from signalflow.detector.local_extrema import LocalExtremaDetector

# Market-wide detectors
from signalflow.detector.market import (
    AgreementDetector,
    CusumEventDetector,
    # Backward compatibility aliases
    GlobalEventDetector,
    MarketCusumDetector,
    MarketZScoreDetector,
    ZScoreEventDetector,
)
from signalflow.detector.percentile_regime import PercentileRegimeDetector
from signalflow.detector.sma_cross import ExampleSmaCrossDetector
from signalflow.detector.structure_detector import StructureDetector
from signalflow.detector.volatility_detector import VolatilityDetector

# Generic detectors
from signalflow.detector.zscore_anomaly import ZScoreAnomalyDetector

__all__ = [
    # Base
    "SignalDetector",
    "ExampleSmaCrossDetector",
    # Market-wide detectors
    "AgreementDetector",
    "MarketZScoreDetector",
    "MarketCusumDetector",
    # Backward compat aliases for market-wide
    "GlobalEventDetector",
    "ZScoreEventDetector",
    "CusumEventDetector",
    # Preset detectors (backward compat)
    "AnomalyDetector",
    "VolatilityDetector",
    "StructureDetector",
    # Generic detectors
    "ZScoreAnomalyDetector",
    "PercentileRegimeDetector",
    "LocalExtremaDetector",
    # Funding rate
    "FundingRateDetector",
]
