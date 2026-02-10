"""Market-wide signal detectors.

Detectors for identifying timestamps where exogenous market-wide events occur
(e.g., flash crashes, regulatory announcements, coordinated moves across pairs).

All detectors extend SignalDetector with signal_category=MARKET_WIDE.
For masking labels during detected signals, use mask_targets_by_signals()
from signalflow.target.utils.

Detectors:
    AgreementDetector: Detects high cross-pair return agreement.
    MarketZScoreDetector: Detects z-score outliers in aggregate return.
    MarketCusumDetector: Detects regime shifts via CUSUM.

Example:
    ```python
    from signalflow.detector.market import AgreementDetector
    from signalflow.target.utils import mask_targets_by_signals

    detector = AgreementDetector(agreement_threshold=0.8)
    signals = detector.run(raw_data_view)

    # Mask training labels around detected signals
    labeled_df = mask_targets_by_signals(
        df=labeled_df,
        signals=signals,
        mask_signal_types={"market_agreement"},
        horizon_bars=60,
    )
    ```
"""

from signalflow.detector.market.agreement_detector import AgreementDetector, GlobalEventDetector
from signalflow.detector.market.zscore_detector import MarketZScoreDetector, ZScoreEventDetector
from signalflow.detector.market.cusum_detector import MarketCusumDetector, CusumEventDetector

__all__ = [
    # New names
    "AgreementDetector",
    "MarketZScoreDetector",
    "MarketCusumDetector",
    # Backward compatibility aliases
    "GlobalEventDetector",
    "ZScoreEventDetector",
    "CusumEventDetector",
]
