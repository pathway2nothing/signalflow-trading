"""Real-time anomaly detector for extreme price movement events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from signalflow.core import RawDataView, Signals, sf_component
from signalflow.core.enums import SignalCategory
from signalflow.detector.base import SignalDetector


@dataclass
@sf_component(name="anomaly_detector")
class AnomalyDetector(SignalDetector):
    """Detects anomalous price movements in real-time (backward-looking only).

    Unlike ``AnomalyLabeler``, this detector uses only past data and is safe
    for live trading. It flags the current bar as anomalous when the current
    return exceeds a multiple of rolling volatility.

    Algorithm:
        1. Compute log returns: log(close[t] / close[t-1])
        2. Compute rolling std of returns over ``vol_window`` bars
        3. Current bar return magnitude: |log_return[t]|
        4. If magnitude > threshold_return_std * rolling_std[t] -> "extreme_positive_anomaly"
        5. If magnitude > threshold AND return is negative -> "extreme_negative_anomaly"
        6. Otherwise: row is skipped (no signal emitted)

    Attributes:
        price_col (str): Price column name. Default: "close".
        vol_window (int): Rolling window for volatility estimation. Default: 1440.
        threshold_return_std (float): Number of standard deviations for anomaly
            threshold. Default: 4.0.

    Example:
        ```python
        from signalflow.core import RawData, RawDataView
        from signalflow.detector.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(
            vol_window=1440,
            threshold_return_std=4.0,
        )
        signals = detector.run(raw_data_view)
        ```

    Note:
        This detector overrides ``preprocess()`` to work directly with raw
        OHLCV data and does not require a FeaturePipeline.
    """

    signal_category: SignalCategory = SignalCategory.ANOMALY
    allowed_signal_types: set[str] | None = field(
        default_factory=lambda: {"extreme_positive_anomaly", "extreme_negative_anomaly"}
    )

    price_col: str = "close"
    vol_window: int = 1440
    threshold_return_std: float = 4.0

    def preprocess(
        self,
        raw_data_view: RawDataView,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Extract raw OHLCV data without feature pipeline.

        Overrides base ``preprocess()`` to bypass FeaturePipeline and return
        the raw spot data directly.

        Args:
            raw_data_view (RawDataView): View to raw market data.
            context (dict[str, Any] | None): Additional context (unused).

        Returns:
            pl.DataFrame: Raw OHLCV data sorted by (pair, timestamp).
        """
        key = self.raw_data_type.value if hasattr(self.raw_data_type, "value") else str(self.raw_data_type)
        return raw_data_view.to_polars(key).sort([self.pair_col, self.ts_col])

    def detect(
        self,
        features: pl.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> Signals:
        """Detect anomalous price movements on the current bar.

        Args:
            features (pl.DataFrame): OHLCV data with pair and timestamp columns.
            context (dict[str, Any] | None): Additional context (unused).

        Returns:
            Signals: Detected anomaly signals with columns:
                pair, timestamp, signal_type, signal, probability.
        """
        price = pl.col(self.price_col)

        # Step 1: log returns (per pair)
        df = features.with_columns(
            (price / price.shift(1).over(self.pair_col)).log().alias("_log_ret"),
        )

        # Step 2: rolling std of returns (per pair)
        df = df.with_columns(
            pl.col("_log_ret")
            .rolling_std(window_size=self.vol_window, min_samples=max(2, self.vol_window // 4))
            .over(self.pair_col)
            .alias("_rolling_vol"),
        )

        # Step 3: current bar return magnitude
        df = df.with_columns(
            pl.col("_log_ret").abs().alias("_ret_abs"),
        )

        # Step 4-5: classify
        threshold = pl.col("_rolling_vol") * self.threshold_return_std

        is_anomaly = (
            pl.col("_ret_abs").is_not_null()
            & pl.col("_rolling_vol").is_not_null()
            & (pl.col("_rolling_vol") > 0)
            & (pl.col("_ret_abs") > threshold)
        )

        is_flash_crash = is_anomaly & (pl.col("_log_ret") < 0)

        signal_type_expr = (
            pl.when(is_flash_crash)
            .then(pl.lit("extreme_negative_anomaly"))
            .when(is_anomaly)
            .then(pl.lit("extreme_positive_anomaly"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("signal_type")
        )

        # Compute probability as ratio of return magnitude to threshold
        probability_expr = (
            pl.when(is_anomaly)
            .then((pl.col("_ret_abs") / threshold).clip(0.0, 1.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("probability")
        )

        df = df.with_columns([signal_type_expr, probability_expr])

        # Step 6: filter to only anomalous bars (skip rows with no signal)
        signals_df = df.filter(pl.col("signal_type").is_not_null()).select(
            [
                self.pair_col,
                self.ts_col,
                "signal_type",
                pl.lit(1).alias("signal"),
                "probability",
            ]
        )

        return Signals(signals_df)
