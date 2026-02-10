"""Feature informativeness analysis via Mutual Information.

Measures how informative each feature is relative to multiple targets
at multiple prediction horizons. Combines MI magnitude with temporal
stability into a composite score.

Example:
    ```python
    from signalflow.feature.informativeness import FeatureInformativenessAnalyzer

    analyzer = FeatureInformativenessAnalyzer()
    report = analyzer.analyze(df_with_features, ["rsi_14", "sma_20", "volume_ratio"])
    print(report.top_features(10))
    print(report.score_matrix)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from signalflow.feature.mutual_information import (
    entropy_continuous,
    entropy_discrete,
    mutual_information_continuous,
    mutual_information_continuous_discrete,
    mutual_information_discrete,
    normalized_mutual_information,
)
from signalflow.target.multi_target_generator import (
    DEFAULT_HORIZONS,
    DEFAULT_TARGET_TYPES,
    HorizonConfig,
    MultiTargetGenerator,
    TargetType,
)
from signalflow.target.utils import mask_targets_by_signals

if TYPE_CHECKING:
    from signalflow.core import RawDataView
    from signalflow.detector.base import SignalDetector


def _default_event_detector() -> "SignalDetector":
    """Deferred import to avoid circular dependency."""
    from signalflow.detector.market import AgreementDetector

    return AgreementDetector()


def _df_to_raw_data_view(
    df: pl.DataFrame,
    pair_col: str = "pair",
    ts_col: str = "timestamp",
) -> "RawDataView":
    """Convert a DataFrame to RawDataView for detector usage.

    Creates a minimal RawDataView from an OHLCV DataFrame.

    Args:
        df: DataFrame with OHLCV columns.
        pair_col: Pair column name.
        ts_col: Timestamp column name.

    Returns:
        RawDataView wrapping the DataFrame.
    """
    from signalflow.core import RawData, RawDataView

    pairs = df.get_column(pair_col).unique().sort().to_list()
    timestamps = df.get_column(ts_col).sort()
    datetime_start = timestamps.min()
    datetime_end = timestamps.max()

    raw_data = RawData(
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        pairs=pairs,
        data={"spot": df},
    )
    return RawDataView(raw_data)


@dataclass
class RollingMIConfig:
    """Configuration for rolling MI stability computation.

    Attributes:
        window_size: Number of bars per rolling window.
        min_window_fill: Minimum fraction of non-null values in a window.
    """

    window_size: int = 5000
    min_window_fill: float = 0.7


@dataclass
class CompositeWeights:
    """Weights for composite informativeness scoring.

    Attributes:
        horizon_weights: Per-horizon weights. None = equal weights.
        target_weights: Per-target weights. None = equal weights.
        stability_weight: Fraction of score from stability (rest from NMI).
    """

    horizon_weights: dict[str, float] | None = None
    target_weights: dict[str, float] | None = None
    stability_weight: float = 0.3


@dataclass
class InformativenessReport:
    """Container for informativeness analysis results.

    Attributes:
        raw_mi: Full MI results (feature × horizon × target).
        composite_scores: Aggregated scores per feature, ranked.
        score_matrix: Pivoted Feature × (Horizon, Target) matrix.
        global_events: Global event detection results (if enabled).
        metadata: Analysis configuration and statistics.
    """

    raw_mi: pl.DataFrame
    composite_scores: pl.DataFrame
    score_matrix: pl.DataFrame
    global_events: pl.DataFrame | None
    metadata: dict

    def top_features(self, n: int = 20) -> pl.DataFrame:
        """Return top N features by composite score."""
        return self.composite_scores.head(n)

    def bottom_features(self, n: int = 20) -> pl.DataFrame:
        """Return bottom N features (least informative)."""
        return self.composite_scores.tail(n)

    def feature_detail(self, feature_name: str) -> pl.DataFrame:
        """Return detailed MI breakdown for a single feature."""
        return self.raw_mi.filter(pl.col("feature") == feature_name)


@dataclass
class FeatureInformativenessAnalyzer:
    """Orchestrator for feature informativeness analysis.

    Pipeline:
        1. Generate multi-horizon, multi-target labels
        2. Detect and mask global events
        3. Compute MI between each feature and each target
        4. Compute rolling MI for temporal stability
        5. Compute composite weighted scores
        6. Generate report

    Attributes:
        target_generator: Multi-target label generator.
        event_detector: Global event detector. None to disable.
        rolling_mi: Rolling MI stability configuration.
        weights: Composite scoring weights.
        bins: Number of histogram bins for MI estimation.
        pair_col: Pair column name.
        ts_col: Timestamp column name.
        aggregate_pairs: If True, pool all pairs for MI computation.
    """

    target_generator: MultiTargetGenerator = field(default_factory=MultiTargetGenerator)
    event_detector: SignalDetector | None = field(default_factory=_default_event_detector)
    rolling_mi: RollingMIConfig = field(default_factory=RollingMIConfig)
    weights: CompositeWeights = field(default_factory=CompositeWeights)
    bins: int = 20
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    aggregate_pairs: bool = True

    def analyze(
        self,
        df: pl.DataFrame,
        feature_columns: list[str],
    ) -> InformativenessReport:
        """Run full informativeness analysis.

        Args:
            df: OHLCV DataFrame with pre-computed feature columns.
            feature_columns: List of feature column names to evaluate.

        Returns:
            InformativenessReport with all results.

        Raises:
            ValueError: If required columns are missing or feature_columns is empty.
        """
        self._validate(df, feature_columns)

        # 1. Generate targets
        logger.info("Generating multi-horizon targets...")
        df = self.target_generator.generate(df)
        target_meta = self.target_generator.target_columns()

        # 2. Detect and mask global events
        global_events = None
        if self.event_detector is not None:
            logger.info("Detecting global events...")
            # Convert DataFrame to RawDataView for SignalDetector
            raw_view = _df_to_raw_data_view(df, self.pair_col, self.ts_col)
            signals = self.event_detector.run(raw_view)
            global_events = signals.value

            # Get all target columns
            target_columns = [meta["column"] for meta in target_meta]

            # Mask targets using the maximum horizon
            max_horizon = max(h.horizon for h in self.target_generator.horizons)

            df = mask_targets_by_signals(
                df=df,
                signals=signals,
                mask_signal_types=self.event_detector.allowed_signal_types or set(),
                horizon_bars=max_horizon,
                cooldown_bars=60,
                target_columns=target_columns,
                pair_col=self.pair_col,
                ts_col=self.ts_col,
            )

        # 3-4. Compute MI and rolling stability
        logger.info(f"Computing MI for {len(feature_columns)} features × {len(target_meta)} targets...")
        mi_rows = self._compute_all_mi(df, feature_columns, target_meta)
        raw_mi = pl.DataFrame(mi_rows)

        # 5. Composite scoring
        logger.info("Computing composite scores...")
        composite_scores = self._compute_composite(raw_mi)
        score_matrix = self._build_score_matrix(raw_mi)

        n_events = 0
        if global_events is not None:
            n_events = global_events.height

        metadata = {
            "n_features": len(feature_columns),
            "n_horizons": len(self.target_generator.horizons),
            "n_target_types": len(self.target_generator.target_types),
            "n_global_events": n_events,
            "bins": self.bins,
            "aggregate_pairs": self.aggregate_pairs,
        }

        logger.info("Informativeness analysis complete.")
        return InformativenessReport(
            raw_mi=raw_mi,
            composite_scores=composite_scores,
            score_matrix=score_matrix,
            global_events=global_events,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # MI computation
    # ------------------------------------------------------------------

    def _compute_all_mi(
        self,
        df: pl.DataFrame,
        feature_columns: list[str],
        target_meta: list[dict[str, str]],
    ) -> list[dict]:
        """Compute MI for all (feature, target) pairs."""
        rows = []

        for feat_col in feature_columns:
            for tmeta in target_meta:
                target_col = tmeta["column"]
                target_kind = tmeta["kind"]

                feat_arr, target_arr = self._extract_arrays(df, feat_col, target_col)
                if feat_arr is None:
                    rows.append(self._nan_row(feat_col, tmeta))
                    continue

                mi = self._compute_mi_pair(feat_arr, target_arr, target_kind)
                h_feat = entropy_continuous(feat_arr, bins=self.bins)
                h_target = (
                    entropy_discrete(target_arr)
                    if target_kind == "discrete"
                    else entropy_continuous(target_arr, bins=self.bins)
                )
                nmi = normalized_mutual_information(mi, h_feat, h_target)

                stability = self._compute_stability(feat_arr, target_arr, target_kind)

                rows.append(
                    {
                        "feature": feat_col,
                        "horizon": tmeta["horizon"],
                        "target_type": tmeta["target_type"],
                        "mi": mi,
                        "nmi": nmi,
                        "stability_score": stability,
                    }
                )

        return rows

    def _extract_arrays(
        self,
        df: pl.DataFrame,
        feat_col: str,
        target_col: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract aligned numpy arrays, dropping rows with nulls in either."""
        if feat_col not in df.columns or target_col not in df.columns:
            return None, None

        subset = df.select([feat_col, target_col]).drop_nulls()
        if subset.height < 10:
            return None, None

        feat_arr = subset.get_column(feat_col).to_numpy().astype(np.float64)
        target_series = subset.get_column(target_col)

        if target_series.dtype == pl.Utf8:
            target_arr = target_series.to_numpy()
        else:
            target_arr = target_series.to_numpy().astype(np.float64)

        return feat_arr, target_arr

    def _compute_mi_pair(
        self,
        feat: np.ndarray,
        target: np.ndarray,
        target_kind: str,
    ) -> float:
        """Compute MI between one feature and one target."""
        if target_kind == "discrete":
            return mutual_information_continuous_discrete(feat, target, bins=self.bins)
        return mutual_information_continuous(feat, target, bins=self.bins)

    def _compute_stability(
        self,
        feat: np.ndarray,
        target: np.ndarray,
        target_kind: str,
    ) -> float:
        """Compute temporal stability via rolling MI windows."""
        cfg = self.rolling_mi
        n = len(feat)
        step = cfg.window_size
        min_fill = int(step * cfg.min_window_fill)

        mi_values = []
        for start in range(0, n - min_fill + 1, step):
            end = min(start + step, n)
            f_win = feat[start:end]
            t_win = target[start:end]

            # Check fill rate
            if np.issubdtype(f_win.dtype, np.floating):
                valid = np.isfinite(f_win).sum()
            else:
                valid = len(f_win)

            if valid < min_fill:
                continue

            mi = self._compute_mi_pair(f_win, t_win, target_kind)
            if not np.isnan(mi):
                mi_values.append(mi)

        if len(mi_values) < 2:
            return np.nan

        mean_mi = np.mean(mi_values)
        std_mi = np.std(mi_values)

        if mean_mi <= 0:
            return 0.0

        cv = std_mi / mean_mi
        return 1.0 / (1.0 + cv)

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def _compute_composite(self, raw_mi: pl.DataFrame) -> pl.DataFrame:
        """Compute composite scores from raw MI results."""
        if raw_mi.height == 0:
            return pl.DataFrame(schema={"feature": pl.Utf8, "composite_score": pl.Float64, "rank": pl.UInt32})

        w = self.weights
        alpha = w.stability_weight

        # Build weights per (horizon, target_type)
        horizons = raw_mi.get_column("horizon").unique().to_list()
        targets = raw_mi.get_column("target_type").unique().to_list()

        h_weights = w.horizon_weights or {h: 1.0 / len(horizons) for h in horizons}
        t_weights = w.target_weights or {t: 1.0 / len(targets) for t in targets}

        # Normalize
        h_total = sum(h_weights.values())
        t_total = sum(t_weights.values())
        h_weights = {k: v / h_total for k, v in h_weights.items()}
        t_weights = {k: v / t_total for k, v in t_weights.items()}

        scored = (
            raw_mi.with_columns(
                [
                    pl.col("horizon").replace_strict(h_weights, default=0.0).alias("_h_w"),
                    pl.col("target_type").replace_strict(t_weights, default=0.0).alias("_t_w"),
                ]
            )
            .with_columns((pl.col("_h_w") * pl.col("_t_w")).alias("_weight"))
            .with_columns(
                ((1.0 - alpha) * pl.col("nmi").fill_null(0.0) + alpha * pl.col("stability_score").fill_null(0.0)).alias(
                    "_cell_score"
                )
            )
            .with_columns((pl.col("_cell_score") * pl.col("_weight")).alias("_weighted_score"))
        )

        result = (
            scored.group_by("feature")
            .agg(pl.col("_weighted_score").sum().alias("composite_score"))
            .sort("composite_score", descending=True)
            .with_row_index("rank", offset=1)
            .select(["feature", "composite_score", "rank"])
        )

        return result

    def _build_score_matrix(self, raw_mi: pl.DataFrame) -> pl.DataFrame:
        """Build pivoted Feature × (Horizon, Target) matrix."""
        if raw_mi.height == 0:
            return pl.DataFrame()

        matrix = raw_mi.with_columns((pl.col("horizon") + "_" + pl.col("target_type")).alias("_col_key")).pivot(
            on="_col_key",
            index="feature",
            values="nmi",
        )

        return matrix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nan_row(self, feat_col: str, tmeta: dict) -> dict:
        return {
            "feature": feat_col,
            "horizon": tmeta["horizon"],
            "target_type": tmeta["target_type"],
            "mi": np.nan,
            "nmi": np.nan,
            "stability_score": np.nan,
        }

    def _validate(self, df: pl.DataFrame, feature_columns: list[str]) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")

        required = {self.pair_col, self.ts_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")
