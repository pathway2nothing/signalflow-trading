from signalflow.detector.base import SignalDetector
from signalflow.detector.sma_cross import ExampleSmaCrossDetector

# Market-wide detectors
from signalflow.detector.market import (
    AgreementDetector,
    MarketZScoreDetector,
    MarketCusumDetector,
    # Backward compatibility aliases
    GlobalEventDetector,
    ZScoreEventDetector,
    CusumEventDetector,
)

# Preset detectors (backward compat)
from signalflow.detector.anomaly_detector import AnomalyDetector
from signalflow.detector.volatility_detector import VolatilityDetector
from signalflow.detector.structure_detector import StructureDetector

# Generic detectors
from signalflow.detector.zscore_anomaly import ZScoreAnomalyDetector
from signalflow.detector.percentile_regime import PercentileRegimeDetector
from signalflow.detector.local_extrema import LocalExtremaDetector

# Funding rate detector
from signalflow.detector.funding_rate import FundingRateDetector

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
