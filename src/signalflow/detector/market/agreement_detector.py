"""Agreement-based market-wide detector.

Detects timestamps where an unusually high fraction of trading pairs
move in the same direction simultaneously, signaling an exogenous
macro event (interest rate decision, regulatory news, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl
from loguru import logger

from signalflow.core import RawDataView, Signals
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SignalCategory, SfComponentType
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="market_wide/agreement")
class AgreementDetector(SignalDetector):
    """Detects timestamps where cross-pair return agreement is abnormally high.

    A market-wide detector that signals when a high fraction of trading
    pairs move in the same direction simultaneously. This indicates exogenous
    macro events (interest rate decisions, regulatory news, etc.) that cannot
    be predicted from individual pair features.

    Algorithm:
        1. Compute log-return for each pair at each timestamp.
        2. At each timestamp, compute the fraction of pairs with
           same-sign return (majority sign).
        3. If fraction >= ``agreement_threshold``, emit signal.

    Attributes:
        agreement_threshold: Fraction of pairs that must agree for detection.
        min_pairs: Minimum number of active pairs at a timestamp.
        return_window: Bars for return computation.
        signal_type_name: Signal type name for detected signals.

    Example:
        ```python
        from signalflow.detector import AgreementDetector
        from signalflow.target.utils import mask_targets_by_signals

        # Detect market-wide agreement
        detector = AgreementDetector(agreement_threshold=0.8)
        signals = detector.run(raw_data_view)

        # Mask labels overlapping with detected signals
        labeled_df = mask_targets_by_signals(
            df=labeled_df,
            signals=signals,
            mask_signal_types={"market_agreement"},
            horizon_bars=60,
        )
        ```

    Note:
        Returns one signal per timestamp (not per pair) since market-wide
        signals affect all pairs simultaneously. The signal has a synthetic
        "ALL" pair.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    signal_category: SignalCategory = SignalCategory.MARKET_WIDE
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"market_agreement"})

    agreement_threshold: float = 0.8
    min_pairs: int = 5
    return_window: int = 1
    price_col: str = "close"
    signal_type_name: str = "market_agreement"

    def __post_init__(self) -> None:
        self.allowed_signal_types = {self.signal_type_name}

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Preprocess raw data: compute log returns.

        Returns raw OHLCV with _ret column added.
        """
        df = super().preprocess(raw_data_view, context)

        df = df.sort([self.pair_col, self.ts_col]).with_columns(
            (pl.col(self.price_col) / pl.col(self.price_col).shift(self.return_window))
            .log()
            .over(self.pair_col)
            .alias("_ret")
        )

        return df

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect market-wide agreement timestamps.

        Args:
            features: Multi-pair OHLCV DataFrame with _ret column.
            context: Additional context (unused).

        Returns:
            Signals with market_agreement signal type for detected timestamps.
        """
        agreement = (
            features.filter(pl.col("_ret").is_not_null() & pl.col("_ret").is_finite())
            .group_by(self.ts_col)
            .agg(
                pl.col("_ret").count().alias("_n_pairs"),
                (pl.col("_ret") > 0).sum().alias("_n_positive"),
                (pl.col("_ret") < 0).sum().alias("_n_negative"),
            )
            .filter(pl.col("_n_pairs") >= self.min_pairs)
            .with_columns(
                (
                    pl.max_horizontal("_n_positive", "_n_negative").cast(pl.Float64)
                    / pl.col("_n_pairs").cast(pl.Float64)
                ).alias("_agreement")
            )
            .filter(pl.col("_agreement") >= self.agreement_threshold)
        )

        n_signals = agreement.height
        logger.info(f"AgreementDetector: detected {n_signals} timestamps")

        if n_signals == 0:
            return Signals(
                pl.DataFrame(
                    schema={
                        self.pair_col: pl.Utf8,
                        self.ts_col: pl.Datetime,
                        "signal_type": pl.Utf8,
                        "signal": pl.Int64,
                        "probability": pl.Float64,
                    }
                )
            )

        # Create signals with synthetic "ALL" pair for market-wide signals
        signals_df = agreement.select(
            [
                pl.lit("ALL").alias(self.pair_col),
                self.ts_col,
                pl.lit(self.signal_type_name).alias("signal_type"),
                pl.lit(1).alias("signal"),
                pl.col("_agreement").alias("probability"),
            ]
        )

        return Signals(signals_df)


# Backward compatibility alias with legacy default signal_type_name
@dataclass
class GlobalEventDetector(AgreementDetector):
    """Backward compatibility alias for AgreementDetector.

    .. deprecated::
        Use AgreementDetector instead. This alias preserves the legacy
        "global_event" signal_type_name for backward compatibility.
    """

    signal_type_name: str = "global_event"
    allowed_signal_types: set[str] | None = field(default_factory=lambda: {"global_event"})
