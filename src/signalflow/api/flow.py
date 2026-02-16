"""
FlowBuilder - Unified pipeline API for SignalFlow.

Flow is a graph of nodes where the output depends on:
1. The structure of the graph (which nodes are present)
2. Metric nodes (which metrics to compute)

Examples:
    Data → Features → FeatureMetrics         = Feature analysis
    Data → Detector → SignalMetrics          = Signal analysis
    Data → Detector → Labeler → Validator    = Validation metrics
    Data → ... → Strategy → BacktestMetrics  = Full backtest
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Callable, Self

import polars as pl

from signalflow.core import (
    RawData,
    Signals,
    SfComponentType,
    default_registry,
)
from signalflow.api.exceptions import (
    ConfigurationError,
    MissingDataError,
)

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.validator.base import SignalValidator
    from signalflow.feature import FeaturePipeline
    from signalflow.target.base import SignalLabeler


class RunMode(str, Enum):
    """Flow execution modes."""

    SINGLE = "single"
    TEMPORAL_CV = "temporal_cv"
    WALK_FORWARD = "walk_forward"
    LIVE = "live"


class AggregationMode(str, Enum):
    """Signal aggregation modes for multiple detectors."""

    MERGE = "merge"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    ANY = "any"
    META_LABELING = "meta_labeling"


@dataclass
class FlowConfig:
    """Configuration snapshot of a Flow."""

    strategy_id: str
    run_mode: RunMode
    data_sources: list[str]
    feature_pipelines: list[str]
    detectors: list[str]
    labelers: list[str]
    validators: list[str]
    metrics: list[str]
    capital: float
    fee: float


@dataclass
class FlowResult:
    """Container for Flow execution results.

    Attributes are populated based on which metric nodes were configured.
    """

    # Metrics (populated based on metric nodes)
    feature_metrics: Any | None = None
    signal_metrics: Any | None = None
    label_metrics: Any | None = None
    validation_metrics: Any | None = None
    backtest_metrics: Any | None = None
    live_metrics: Any | None = None

    # Intermediate data (for debugging/analysis)
    features: pl.DataFrame | None = None
    signals: Signals | None = None
    labels: pl.DataFrame | None = None
    predictions: pl.DataFrame | None = None
    trades: pl.DataFrame | None = None

    # Metadata
    flow_config: FlowConfig | None = None
    execution_time: float = 0.0

    # For walk-forward mode
    fold_results: list["FlowResult"] | None = None
    window_results: list["FlowResult"] | None = None

    def summary(self) -> str:
        """Generate unified summary across all metrics."""
        lines = ["=" * 60, "FLOW RESULT SUMMARY", "=" * 60]

        if self.backtest_metrics:
            lines.append("\nBacktest Metrics:")
            for k, v in self._format_metrics(self.backtest_metrics).items():
                lines.append(f"  {k}: {v}")

        if self.validation_metrics:
            lines.append("\nValidation Metrics:")
            for k, v in self._format_metrics(self.validation_metrics).items():
                lines.append(f"  {k}: {v}")

        if self.signal_metrics:
            lines.append("\nSignal Metrics:")
            for k, v in self._format_metrics(self.signal_metrics).items():
                lines.append(f"  {k}: {v}")

        if self.feature_metrics:
            lines.append("\nFeature Metrics:")
            for k, v in self._format_metrics(self.feature_metrics).items():
                lines.append(f"  {k}: {v}")

        lines.append(f"\nExecution time: {self.execution_time:.2f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_metrics(self, metrics: Any) -> dict[str, str]:
        """Format metrics for display."""
        if isinstance(metrics, dict):
            return {k: self._format_value(v) for k, v in metrics.items()}
        if hasattr(metrics, "__dict__"):
            return {k: self._format_value(v) for k, v in metrics.__dict__.items() if not k.startswith("_")}
        return {"value": str(metrics)}

    @staticmethod
    def _format_value(v: Any) -> str:
        if isinstance(v, float):
            if abs(v) < 1:
                return f"{v:.4f}"
            return f"{v:,.2f}"
        return str(v)

    def save_artifacts(self, path: str | Path) -> None:
        """Save intermediate artifacts to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.features is not None:
            self.features.write_parquet(path / "features.parquet")
        if self.signals is not None:
            self.signals.value.write_parquet(path / "signals.parquet")
        if self.labels is not None:
            self.labels.write_parquet(path / "labels.parquet")
        if self.predictions is not None:
            self.predictions.write_parquet(path / "predictions.parquet")
        if self.trades is not None:
            self.trades.write_parquet(path / "trades.parquet")


@dataclass
class FlowBuilder:
    """
    Fluent builder for unified SignalFlow pipelines.

    FlowBuilder extends BacktestBuilder with:
    - Feature pipelines (.features())
    - Target labeling (.labeler())
    - Metric nodes (.metrics())
    - Multiple run modes (.run(mode=...))
    - Artifact saving

    Example (feature analysis):
        >>> result = (
        ...     sf.flow()
        ...     .data(store="binance", pair="BTC/USDT", timeframe="1h")
        ...     .features(pipeline="momentum")
        ...     .metrics(sf.FeatureMetrics())
        ...     .run()
        ... )

    Example (full backtest):
        >>> result = (
        ...     sf.flow()
        ...     .data(store="binance", pair="BTC/USDT")
        ...     .features(pipeline="full")
        ...     .detector("rsi_cross")
        ...     .labeler("triple_barrier", tp=0.02, sl=0.01)
        ...     .validator("lightgbm")
        ...     .entry("market")
        ...     .exit(tp=0.02, sl=0.01)
        ...     .metrics(sf.BacktestMetrics())
        ...     .run()
        ... )

    Example (walk-forward validation):
        >>> result = (
        ...     sf.flow()
        ...     .data(...)
        ...     .detector(...)
        ...     .validator(...)
        ...     .run(
        ...         mode="walk_forward",
        ...         train_size="6M",
        ...         test_size="1M",
        ...         step="1M",
        ...     )
        ... )
    """

    strategy_id: str = "flow"

    # Data
    _raw: RawData | None = field(default=None, repr=False)
    _named_data: dict[str, RawData | dict[str, Any]] = field(default_factory=dict, repr=False)

    # Features
    _feature_pipelines: dict[str, "FeaturePipeline"] = field(default_factory=dict, repr=False)

    # Detection
    _named_detectors: dict[str, "SignalDetector"] = field(default_factory=dict, repr=False)
    _signals: Signals | None = field(default=None, repr=False)
    _aggregation_config: dict[str, Any] | None = field(default=None, repr=False)

    # Labeling
    _named_labelers: dict[str, "SignalLabeler"] = field(default_factory=dict, repr=False)

    # Validation
    _named_validators: dict[str, "SignalValidator"] = field(default_factory=dict, repr=False)

    # Strategy
    _entry_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _exit_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _capital: float = 10_000.0
    _fee: float = 0.001

    # Metrics
    _metric_nodes: list[Any] = field(default_factory=list, repr=False)

    # Artifacts
    _artifacts_dir: Path | None = field(default=None, repr=False)

    # =========================================================================
    # Data Configuration
    # =========================================================================

    def data(
        self,
        raw: RawData | None = None,
        *,
        name: str | None = None,
        store: str | Path | None = None,
        pair: str | None = None,
        pairs: list[str] | None = None,
        timeframe: str = "1m",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> Self:
        """
        Add a data source to the flow.

        Args:
            raw: Pre-loaded RawData instance
            name: Unique name for cross-referencing
            store: Path to DuckDB store or exchange name
            pair: Single trading pair
            pairs: Multiple trading pairs
            timeframe: Candle timeframe
            start: Start date
            end: End date

        Returns:
            Self for method chaining
        """
        if name is not None:
            if raw is not None:
                self._named_data[name] = raw
            else:
                self._named_data[name] = {
                    "store": store,
                    "pairs": [pair] if pair else pairs,
                    "timeframe": timeframe,
                    "start": start,
                    "end": end,
                }
        else:
            if raw is not None:
                self._raw = raw
            elif not self._named_data:
                self._named_data["default"] = {
                    "store": store,
                    "pairs": [pair] if pair else pairs,
                    "timeframe": timeframe,
                    "start": start,
                    "end": end,
                }
        return self

    # =========================================================================
    # Feature Configuration
    # =========================================================================

    def features(
        self,
        pipeline: "FeaturePipeline | str",
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a feature pipeline to the flow.

        Args:
            pipeline: FeaturePipeline instance or registry name
            name: Unique name for cross-referencing
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining
        """
        if isinstance(pipeline, str):
            try:
                instance = default_registry.create(
                    SfComponentType.FEATURE,
                    pipeline,
                    **kwargs,
                )
            except KeyError:
                raise ConfigurationError(f"Feature pipeline '{pipeline}' not found in registry") from None
        else:
            instance = pipeline

        if name is None:
            name = f"features_{len(self._feature_pipelines)}" if self._feature_pipelines else "default"

        self._feature_pipelines[name] = instance
        return self

    # =========================================================================
    # Detector Configuration
    # =========================================================================

    def detector(
        self,
        detector: "SignalDetector | str",
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a signal detector to the flow.

        Args:
            detector: SignalDetector instance or registry name
            name: Unique name for cross-referencing
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining
        """
        if isinstance(detector, str):
            try:
                instance = default_registry.create(
                    SfComponentType.DETECTOR,
                    detector,
                    **kwargs,
                )
            except KeyError:
                from signalflow.api.exceptions import DetectorNotFoundError

                available = default_registry.list(SfComponentType.DETECTOR)
                raise DetectorNotFoundError(detector, available) from None
        else:
            instance = detector

        if name is None:
            name = f"detector_{len(self._named_detectors)}" if self._named_detectors else "default"

        self._named_detectors[name] = instance
        return self

    def signals(self, signals: Signals) -> Self:
        """Use pre-computed signals (skip detection)."""
        self._signals = signals
        return self

    def aggregate(
        self,
        mode: str | AggregationMode = AggregationMode.MERGE,
        *,
        min_agreement: float = 0.5,
        weights: list[float] | None = None,
    ) -> Self:
        """
        Configure how signals from multiple detectors are combined.

        Args:
            mode: Aggregation mode
            min_agreement: Minimum agreement fraction for majority voting
            weights: Per-detector weights for weighted mode

        Returns:
            Self for method chaining
        """
        self._aggregation_config = {
            "mode": mode if isinstance(mode, str) else mode.value,
            "min_agreement": min_agreement,
            "weights": weights,
        }
        return self

    # =========================================================================
    # Labeler Configuration
    # =========================================================================

    def labeler(
        self,
        labeler: "SignalLabeler | str",
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a target labeler to the flow.

        Args:
            labeler: SignalLabeler instance or registry name
            name: Unique name for cross-referencing
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining
        """
        if isinstance(labeler, str):
            try:
                instance = default_registry.create(
                    SfComponentType.LABELER,
                    labeler,
                    **kwargs,
                )
            except KeyError:
                from signalflow.api.exceptions import LabelerNotFoundError

                available = default_registry.list(SfComponentType.LABELER)
                raise LabelerNotFoundError(labeler, available) from None
        else:
            instance = labeler

        if name is None:
            name = f"labeler_{len(self._named_labelers)}" if self._named_labelers else "default"

        self._named_labelers[name] = instance
        return self

    # =========================================================================
    # Validator Configuration
    # =========================================================================

    def validator(
        self,
        validator: "SignalValidator | str",
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a signal validator (meta-labeler) to the flow.

        Args:
            validator: SignalValidator instance or registry name
            name: Unique name for cross-referencing
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining
        """
        if isinstance(validator, str):
            try:
                instance = default_registry.create(
                    SfComponentType.VALIDATOR,
                    validator,
                    **kwargs,
                )
            except KeyError:
                from signalflow.api.exceptions import ValidatorNotFoundError

                available = default_registry.list(SfComponentType.VALIDATOR)
                raise ValidatorNotFoundError(validator, available) from None
        else:
            instance = validator

        if name is None:
            name = f"validator_{len(self._named_validators)}" if self._named_validators else "default"

        self._named_validators[name] = instance
        return self

    # =========================================================================
    # Strategy Configuration
    # =========================================================================

    def entry(
        self,
        rule: str | None = None,
        *,
        size: float | None = None,
        size_pct: float | None = None,
        max_positions: int = 10,
        **kwargs: Any,
    ) -> Self:
        """Configure entry rules."""
        self._entry_config = {
            "rule": rule,
            "size": size,
            "size_pct": size_pct,
            "max_positions": max_positions,
            **kwargs,
        }
        return self

    def exit(
        self,
        rule: str | None = None,
        *,
        tp: float | None = None,
        sl: float | None = None,
        trailing: float | None = None,
        **kwargs: Any,
    ) -> Self:
        """Configure exit rules."""
        self._exit_config = {
            "rule": rule,
            "tp": tp,
            "sl": sl,
            "trailing": trailing,
            **kwargs,
        }
        return self

    def capital(self, amount: float) -> Self:
        """Set initial capital."""
        self._capital = amount
        return self

    def fee(self, rate: float) -> Self:
        """Set trading fee rate."""
        self._fee = rate
        return self

    # =========================================================================
    # Metrics Configuration
    # =========================================================================

    def metrics(self, *metric_nodes: Any) -> Self:
        """
        Add metric nodes to the flow.

        Metric nodes determine what outputs the flow produces.
        Multiple metric nodes can be added for comprehensive analysis.

        Args:
            *metric_nodes: Metric instances (FeatureMetrics, SignalMetrics, etc.)

        Returns:
            Self for method chaining
        """
        self._metric_nodes.extend(metric_nodes)
        return self

    # =========================================================================
    # Artifacts Configuration
    # =========================================================================

    def artifacts(self, path: str | Path) -> Self:
        """
        Enable artifact saving to directory.

        Args:
            path: Directory path for artifacts

        Returns:
            Self for method chaining
        """
        self._artifacts_dir = Path(path)
        return self

    # =========================================================================
    # Execution
    # =========================================================================

    def run(
        self,
        *,
        mode: str | RunMode = RunMode.SINGLE,
        # Temporal CV options
        folds: int = 5,
        gap: int = 0,
        # Walk-forward options
        train_size: str | int | None = None,
        test_size: str | int | None = None,
        step: str | int | None = None,
        retrain: bool = True,
        # Live options
        paper: bool = True,
        # General
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
        cancel_event: Event | None = None,
    ) -> FlowResult:
        """
        Execute the flow and return results.

        Args:
            mode: Execution mode (single, temporal_cv, walk_forward, live)
            folds: Number of folds for temporal CV
            gap: Gap between train/test for temporal CV
            train_size: Training window size for walk-forward
            test_size: Test window size for walk-forward
            step: Step size for walk-forward
            retrain: Whether to retrain validator each window
            paper: Paper trading for live mode
            progress_callback: Progress callback function
            cancel_event: Cancel event for graceful stop

        Returns:
            FlowResult with metrics and artifacts
        """
        start_time = time.time()
        run_mode = RunMode(mode) if isinstance(mode, str) else mode

        # Build config snapshot
        config = FlowConfig(
            strategy_id=self.strategy_id,
            run_mode=run_mode,
            data_sources=list(self._named_data.keys()),
            feature_pipelines=list(self._feature_pipelines.keys()),
            detectors=list(self._named_detectors.keys()),
            labelers=list(self._named_labelers.keys()),
            validators=list(self._named_validators.keys()),
            metrics=[type(m).__name__ for m in self._metric_nodes],
            capital=self._capital,
            fee=self._fee,
        )

        # Execute based on mode
        if run_mode == RunMode.SINGLE:
            result = self._run_single(progress_callback, cancel_event)
        elif run_mode == RunMode.TEMPORAL_CV:
            result = self._run_temporal_cv(folds, gap, progress_callback)
        elif run_mode == RunMode.WALK_FORWARD:
            result = self._run_walk_forward(
                train_size, test_size, step, retrain, progress_callback
            )
        elif run_mode == RunMode.LIVE:
            result = self._run_live(paper)
        else:
            raise ConfigurationError(f"Unknown run mode: {run_mode}")

        result.flow_config = config
        result.execution_time = time.time() - start_time

        # Save artifacts if configured
        if self._artifacts_dir:
            result.save_artifacts(self._artifacts_dir)

        return result

    def _run_single(
        self,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None,
        cancel_event: Event | None,
    ) -> FlowResult:
        """Execute single run."""
        result = FlowResult()

        # 1. Resolve data
        raw = self._resolve_data()

        # 2. Compute features
        features_df = None
        if self._feature_pipelines:
            features_df = self._compute_features(raw)
            result.features = features_df

        # 3. Detect signals
        signals = self._resolve_signals(raw)
        result.signals = signals

        # 4. Apply labeling (if configured)
        labels_df = None
        if self._named_labelers:
            labels_df = self._compute_labels(raw, signals)
            result.labels = labels_df

        # 5. Apply validation (if configured)
        if self._named_validators and labels_df is not None and features_df is not None:
            result.predictions = self._apply_validation(signals, features_df, labels_df)

        # 6. Run backtest (if strategy configured)
        if self._entry_config or self._exit_config or not self._metric_nodes:
            backtest_result = self._run_backtest(raw, signals, progress_callback, cancel_event)
            if backtest_result:
                result.trades = backtest_result.trades_df
                result.backtest_metrics = backtest_result.metrics

        # 7. Compute metrics based on metric nodes
        self._compute_metrics(result)

        return result

    def _run_temporal_cv(
        self,
        folds: int,
        gap: int,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None,
    ) -> FlowResult:
        """Execute temporal cross-validation."""
        result = FlowResult()
        result.fold_results = []

        # Resolve data
        raw = self._resolve_data()

        # Get timestamps for splitting
        first_key = next(iter(raw.data.keys()), "spot")
        df = raw.get(first_key)
        timestamps = df.select("timestamp").to_series().sort()
        n = len(timestamps)

        fold_size = n // (folds + 1)

        for fold in range(folds):
            # Calculate train/test indices
            train_end = fold_size * (fold + 2)
            test_start = train_end + gap
            test_end = min(test_start + fold_size, n)

            if test_start >= n:
                break

            train_ts = timestamps[:train_end]
            test_ts = timestamps[test_start:test_end]

            # Filter data
            train_df = df.filter(pl.col("timestamp").is_in(train_ts))
            test_df = df.filter(pl.col("timestamp").is_in(test_ts))

            # Create fold result (simplified for now)
            fold_result = FlowResult()
            fold_result.signals = self._resolve_signals(raw)
            result.fold_results.append(fold_result)

        return result

    def _run_walk_forward(
        self,
        train_size: str | int | None,
        test_size: str | int | None,
        step: str | int | None,
        retrain: bool,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None,
    ) -> FlowResult:
        """Execute walk-forward validation."""
        result = FlowResult()
        result.window_results = []

        # TODO: Implement walk-forward logic
        # For now, run single
        return self._run_single(progress_callback, None)

    def _run_live(self, paper: bool) -> FlowResult:
        """Execute live/paper trading mode."""
        result = FlowResult()

        # TODO: Implement live mode
        raise NotImplementedError("Live mode not yet implemented")

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _resolve_data(self) -> RawData:
        """Resolve data sources."""
        if self._raw is not None:
            return self._raw

        if not self._named_data:
            raise MissingDataError()

        # Load first data source for now
        from signalflow.api.shortcuts import load

        first_name, first_source = next(iter(self._named_data.items()))

        if isinstance(first_source, RawData):
            return first_source
        elif isinstance(first_source, dict):
            params = {k: v for k, v in first_source.items() if v is not None}
            return load(**params)

        raise MissingDataError()

    def _compute_features(self, raw: RawData) -> pl.DataFrame:
        """Compute features from all pipelines."""
        first_key = next(iter(raw.data.keys()), "spot")
        df = raw.get(first_key)

        for _name, pipeline in self._feature_pipelines.items():
            df = pipeline.compute(df)

        return df

    def _resolve_signals(self, raw: RawData) -> Signals:
        """Detect signals from all detectors."""
        if self._signals is not None:
            return self._signals

        if not self._named_detectors:
            # Return empty signals
            return Signals(pl.DataFrame(schema={"timestamp": pl.Datetime, "pair": pl.Utf8, "signal": pl.Int8}))

        from signalflow.core.containers.raw_data_view import RawDataView

        view = RawDataView(raw=raw)
        signals_list = []

        for _name, detector in self._named_detectors.items():
            signals = detector.run(view)
            signals_list.append(signals)

        if len(signals_list) == 1:
            return signals_list[0]

        # Merge signals
        from functools import reduce

        return reduce(lambda a, b: a + b, signals_list)

    def _compute_labels(self, raw: RawData, signals: Signals) -> pl.DataFrame:
        """Compute labels from all labelers."""
        first_key = next(iter(raw.data.keys()), "spot")
        df = raw.get(first_key)

        results = []
        for _name, labeler in self._named_labelers.items():
            labeled = labeler.compute(
                df=df,
                signals=signals,
                data_context={"signal_keys": signals.value.select(["pair", "timestamp"])},
            )
            results.append(labeled)

        if len(results) == 1:
            return results[0]

        # Merge labels
        return pl.concat(results).unique(["pair", "timestamp"])

    def _apply_validation(
        self,
        signals: Signals,
        features_df: pl.DataFrame,
        labels_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply validators to signals."""
        predictions = []

        for _name, validator in self._named_validators.items():
            try:
                validated = validator.validate_signals(signals, features_df)
                predictions.append(validated.value)
            except (NotImplementedError, Exception):
                pass

        if predictions:
            return pl.concat(predictions).unique(["pair", "timestamp"])
        return pl.DataFrame()

    def _run_backtest(
        self,
        raw: RawData,
        signals: Signals,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None,
        cancel_event: Event | None,
    ) -> Any:
        """Run backtest using BacktestBuilder."""
        from signalflow.api.builder import BacktestBuilder

        builder = BacktestBuilder(strategy_id=self.strategy_id)
        builder._raw = raw
        builder._signals = signals
        builder._entry_config = self._entry_config
        builder._exit_config = self._exit_config
        builder._capital = self._capital
        builder._fee = self._fee

        return builder.run(progress_callback=progress_callback, cancel_event=cancel_event)

    def _compute_metrics(self, result: FlowResult) -> None:
        """Compute metrics based on configured metric nodes."""
        for metric_node in self._metric_nodes:
            metric_type = type(metric_node).__name__

            if metric_type == "FeatureMetrics" and result.features is not None:
                result.feature_metrics = self._compute_feature_metrics(result.features, metric_node)
            elif metric_type == "SignalMetrics" and result.signals is not None:
                result.signal_metrics = self._compute_signal_metrics(result.signals, metric_node)
            elif metric_type == "LabelMetrics" and result.labels is not None:
                result.label_metrics = self._compute_label_metrics(result.labels, metric_node)
            elif metric_type == "ValidationMetrics" and result.predictions is not None:
                result.validation_metrics = self._compute_validation_metrics(result.predictions, metric_node)
            elif metric_type == "BacktestMetrics":
                # Already computed in _run_backtest
                pass

    def _compute_feature_metrics(self, features: pl.DataFrame, metric_node: Any) -> dict:
        """Compute feature analysis metrics."""
        return {
            "n_features": len([c for c in features.columns if c not in ("timestamp", "pair", "close", "open", "high", "low", "volume")]),
            "n_rows": features.height,
            "null_pct": features.null_count().sum_horizontal().item() / (features.height * features.width),
        }

    def _compute_signal_metrics(self, signals: Signals, metric_node: Any) -> dict:
        """Compute signal analysis metrics."""
        df = signals.value
        total = df.height
        active = df.filter(pl.col("signal") != 0).height

        return {
            "total_rows": total,
            "active_signals": active,
            "signal_rate": active / total if total > 0 else 0,
        }

    def _compute_label_metrics(self, labels: pl.DataFrame, metric_node: Any) -> dict:
        """Compute label analysis metrics."""
        total = labels.height
        wins = labels.filter(pl.col("label") == 1).height if "label" in labels.columns else 0

        return {
            "total_labels": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0,
        }

    def _compute_validation_metrics(self, predictions: pl.DataFrame, metric_node: Any) -> dict:
        """Compute validation metrics."""
        return {
            "n_predictions": predictions.height,
        }

    def __repr__(self) -> str:
        parts = [f"strategy_id={self.strategy_id!r}"]
        if self._feature_pipelines:
            parts.append(f"features={list(self._feature_pipelines.keys())}")
        if self._named_detectors:
            parts.append(f"detectors={list(self._named_detectors.keys())}")
        if self._named_labelers:
            parts.append(f"labelers={list(self._named_labelers.keys())}")
        if self._named_validators:
            parts.append(f"validators={list(self._named_validators.keys())}")
        if self._metric_nodes:
            parts.append(f"metrics={[type(m).__name__ for m in self._metric_nodes]}")
        return f"FlowBuilder({', '.join(parts)})"


def flow(strategy_id: str = "flow") -> FlowBuilder:
    """
    Create a new flow builder.

    This is the unified API for SignalFlow pipelines.
    Flow determines output based on the structure of the pipeline
    and configured metric nodes.

    Args:
        strategy_id: Unique identifier for the flow

    Returns:
        FlowBuilder instance for fluent configuration

    Example:
        >>> result = (
        ...     sf.flow()
        ...     .data(raw=my_data)
        ...     .detector("sma_cross")
        ...     .run()
        ... )
    """
    return FlowBuilder(strategy_id=strategy_id)


# Metric node classes
@dataclass
class FeatureMetrics:
    """Metric node for feature analysis."""

    include_correlation: bool = True
    include_importance: bool = True


@dataclass
class SignalMetrics:
    """Metric node for signal analysis."""

    include_frequency: bool = True
    include_clustering: bool = True


@dataclass
class LabelMetrics:
    """Metric node for label analysis."""

    include_distribution: bool = True
    include_holding_time: bool = True


@dataclass
class ValidationMetrics:
    """Metric node for validation analysis."""

    include_confusion_matrix: bool = True
    include_feature_importance: bool = True


@dataclass
class BacktestMetrics:
    """Metric node for full backtest analysis."""

    include_equity_curve: bool = True
    include_drawdown: bool = True


@dataclass
class LiveMetrics:
    """Metric node for live trading analysis."""

    include_latency: bool = True
    include_slippage: bool = True


__all__ = [
    "FlowBuilder",
    "FlowResult",
    "FlowConfig",
    "flow",
    "RunMode",
    "AggregationMode",
    # Metric nodes
    "FeatureMetrics",
    "SignalMetrics",
    "LabelMetrics",
    "ValidationMetrics",
    "BacktestMetrics",
    "LiveMetrics",
]
