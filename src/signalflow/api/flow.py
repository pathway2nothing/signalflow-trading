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

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, ClassVar, Self

import polars as pl

from signalflow.api.exceptions import (
    ConfigurationError,
    MissingDataError,
)
from signalflow.core import (
    RawData,
    SfComponentType,
    Signals,
    default_registry,
    sf_component,
)

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.feature import FeaturePipeline
    from signalflow.target.base import SignalLabeler
    from signalflow.validator.base import SignalValidator


class RunMode(StrEnum):
    """Flow execution modes."""

    SINGLE = "single"
    TEMPORAL_CV = "temporal_cv"
    WALK_FORWARD = "walk_forward"
    LIVE = "live"


class AggregationMode(StrEnum):
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
    fold_results: list[FlowResult] | None = None
    window_results: list[FlowResult] | None = None

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

        # Save flow config
        if self.flow_config is not None:
            config_dict = {
                "strategy_id": self.flow_config.strategy_id,
                "run_mode": self.flow_config.run_mode.value,
                "data_sources": self.flow_config.data_sources,
                "feature_pipelines": self.flow_config.feature_pipelines,
                "detectors": self.flow_config.detectors,
                "labelers": self.flow_config.labelers,
                "validators": self.flow_config.validators,
                "metrics": self.flow_config.metrics,
                "capital": self.flow_config.capital,
                "fee": self.flow_config.fee,
            }
            with open(path / "flow_config.json", "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

        # Save window/fold summaries
        if self.window_results:
            summaries = [wr.to_json_dict() for wr in self.window_results]
            with open(path / "window_results.json", "w") as f:
                json.dump(summaries, f, indent=2, default=str)
        if self.fold_results:
            summaries = [fr.to_json_dict() for fr in self.fold_results]
            with open(path / "fold_results.json", "w") as f:
                json.dump(summaries, f, indent=2, default=str)

    def to_json_dict(self) -> dict[str, Any]:
        """Export as JSON-serializable dict for API responses."""
        result: dict[str, Any] = {}

        if self.backtest_metrics:
            result["metrics"] = self.backtest_metrics if isinstance(self.backtest_metrics, dict) else {}

        result["n_trades"] = self.trades.height if self.trades is not None else 0
        result["execution_time"] = self.execution_time

        if self.trades is not None and self.trades.height > 0:
            result["trades"] = self.trades.to_dicts()

        if self.flow_config:
            result["config"] = {
                "strategy_id": self.flow_config.strategy_id,
                "run_mode": self.flow_config.run_mode.value,
                "capital": self.flow_config.capital,
            }

        if self.window_results:
            result["window_results"] = [wr.to_json_dict() for wr in self.window_results]
        if self.fold_results:
            result["fold_results"] = [fr.to_json_dict() for fr in self.fold_results]

        return result


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
    _feature_pipelines: dict[str, FeaturePipeline] = field(default_factory=dict, repr=False)

    # Detection
    _named_detectors: dict[str, SignalDetector] = field(default_factory=dict, repr=False)
    _signals: Signals | None = field(default=None, repr=False)
    _aggregation_config: dict[str, Any] | None = field(default=None, repr=False)

    # Labeling
    _named_labelers: dict[str, SignalLabeler] = field(default_factory=dict, repr=False)

    # Validation
    _named_validators: dict[str, SignalValidator] = field(default_factory=dict, repr=False)

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
        pipeline: FeaturePipeline | str,
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
        detector: SignalDetector | str,
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
        labeler: SignalLabeler | str,
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
        validator: SignalValidator | str,
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
            result = self._run_walk_forward(train_size, test_size, step, retrain, progress_callback)
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
        """Execute temporal cross-validation.

        Uses expanding window: train grows with each fold, test stays fixed size.
        """
        result = FlowResult()
        result.fold_results = []

        raw = self._resolve_data()

        # Get timestamp boundaries
        first_key = next(iter(raw.data.keys()), "spot")
        df = raw.get(first_key)
        timestamps = df.select("timestamp").to_series().sort()
        n = len(timestamps)

        if n == 0:
            return result

        fold_size = n // (folds + 1)
        if fold_size == 0:
            return result

        timedelta(hours=gap) if gap > 0 else timedelta(0)

        for fold in range(folds):
            if progress_callback:
                progress_callback(fold, folds, {"fold": fold})

            # Expanding train: from start to fold boundary
            train_end_idx = fold_size * (fold + 2)
            test_start_idx = min(train_end_idx + gap, n - 1)
            test_end_idx = min(test_start_idx + fold_size, n)

            if test_start_idx >= n or test_end_idx <= test_start_idx:
                break

            train_start_ts = timestamps[0]
            train_end_ts = timestamps[min(train_end_idx, n - 1)]
            test_start_ts = timestamps[test_start_idx]
            test_end_ts = timestamps[min(test_end_idx - 1, n - 1)]

            fold_result = self._run_on_window(
                raw,
                train_start=train_start_ts,
                train_end=train_end_ts,
                test_start=test_start_ts,
                test_end=test_end_ts,
            )
            result.fold_results.append(fold_result)

        # Aggregate metrics across folds
        if result.fold_results:
            result.backtest_metrics = self._aggregate_results(result.fold_results)

        return result

    def _run_walk_forward(
        self,
        train_size: str | int | None,
        test_size: str | int | None,
        step: str | int | None,
        retrain: bool,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None,
    ) -> FlowResult:
        """Execute walk-forward validation.

        Rolls a train/test window forward through the data, optionally
        retraining the validator each window.
        """
        result = FlowResult()
        result.window_results = []

        train_td = self._parse_period(train_size)
        test_td = self._parse_period(test_size)
        step_td = self._parse_period(step) if step is not None else test_td

        raw = self._resolve_data()

        # Get timestamp boundaries
        first_key = next(iter(raw.data.keys()), "spot")
        df = raw.get(first_key)
        min_ts = df.select("timestamp").min().item()
        max_ts = df.select("timestamp").max().item()

        # Build windows
        windows: list[tuple[datetime, datetime, datetime, datetime]] = []
        train_start = min_ts
        while True:
            train_end = train_start + train_td
            test_start = train_end
            test_end = test_start + test_td
            if test_end > max_ts:
                break
            windows.append((train_start, train_end, test_start, test_end))
            train_start = train_start + step_td

        if not windows:
            raise ConfigurationError(
                f"Cannot create walk-forward windows: data range too short for "
                f"train_size={train_size}, test_size={test_size}. "
                f"Data spans {min_ts} to {max_ts}."
            )

        current_capital = self._capital

        for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            if progress_callback:
                progress_callback(i, len(windows), {"window": i})

            window_result = self._run_on_window(
                raw,
                train_start=tr_start if retrain else None,
                train_end=tr_end if retrain else None,
                test_start=te_start,
                test_end=te_end,
                current_capital=current_capital,
            )
            result.window_results.append(window_result)

            # Compound capital across windows
            if window_result.backtest_metrics and isinstance(window_result.backtest_metrics, dict):
                final_cap = window_result.backtest_metrics.get("final_capital")
                if final_cap is not None and final_cap > 0:
                    current_capital = final_cap

        # Aggregate metrics across windows
        if result.window_results:
            result.backtest_metrics = self._aggregate_results(result.window_results)

        return result

    def _run_live(self, paper: bool) -> FlowResult:
        """Execute live/paper trading mode.

        Wires the FlowBuilder config into a :class:`RealtimeRunner` with
        either a :class:`VirtualRealtimeBroker` (paper) or, in the future,
        a live broker.

        The runner requires ``raw_store`` and ``detector`` at minimum.
        Data sources configured via ``.data(store=...)`` are used to
        locate the DuckDB store.  If a pre-loaded ``RawData`` was given
        instead, live mode raises ``ConfigurationError``.

        Args:
            paper: If True, use VirtualRealtimeBroker (no real money).

        Returns:
            FlowResult with live_metrics populated.
        """
        import asyncio

        from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
        from signalflow.strategy.broker.virtual_broker import VirtualRealtimeBroker
        from signalflow.strategy.runner.realtime_runner import RealtimeRunner

        # --- resolve detector ------------------------------------------------
        if not self._named_detectors:
            raise ConfigurationError("Live mode requires at least one detector")

        detector = next(iter(self._named_detectors.values()))

        # --- resolve raw store -----------------------------------------------
        raw_store, pairs, timeframe = self._resolve_live_store()

        # --- build entry/exit rules ------------------------------------------
        entry_rules, exit_rules = self._build_strategy_rules()

        # --- build metrics ---------------------------------------------------
        from signalflow.analytic.strategy.main_strategy_metrics import (
            DrawdownMetric,
            SharpeRatioMetric,
            TotalReturnMetric,
            WinRateMetric,
        )

        strategy_metrics = [
            TotalReturnMetric(),
            DrawdownMetric(),
            WinRateMetric(),
            SharpeRatioMetric(),
        ]

        # --- build broker ----------------------------------------------------
        store = self._resolve_strategy_store()
        executor = VirtualSpotExecutor(fee_rate=self._fee, slippage_pct=0.0005 if paper else 0.0)
        broker = VirtualRealtimeBroker(executor=executor, store=store)

        # --- optional risk manager -------------------------------------------
        # Users can attach one via broker.risk_manager = ... before calling run

        # --- build runner ----------------------------------------------------
        runner = RealtimeRunner(
            strategy_id=self.strategy_id,
            pairs=pairs,
            timeframe=timeframe,
            initial_capital=self._capital,
            detector=detector,
            broker=broker,
            raw_store=raw_store,
            strategy_store=store,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            metrics=strategy_metrics,
        )

        # --- execute ---------------------------------------------------------
        start = time.time()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an async context — run in a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, runner.run_async()).result()
        else:
            asyncio.run(runner.run_async())

        # --- build result ----------------------------------------------------
        result = FlowResult()
        result.execution_time = time.time() - start

        trades_df = runner.trades_df
        result.trades = trades_df

        metrics_df = runner.metrics_df
        if metrics_df.height > 0:
            last = metrics_df.tail(1)
            result.live_metrics = {col: last.select(col).item() for col in metrics_df.columns if col != "timestamp"}
            result.backtest_metrics = result.live_metrics

        if isinstance(broker, VirtualRealtimeBroker):
            eq_df = broker.equity_curve_df()
            if eq_df.height > 0:
                result.live_metrics = result.live_metrics or {}
                result.live_metrics["equity_curve_length"] = eq_df.height

        return result

    # ---- live-mode helpers --------------------------------------------------

    def _resolve_live_store(self) -> tuple[Any, list[str], str]:
        """Extract raw_store, pairs, timeframe from data config for live mode."""
        if self._raw is not None:
            raise ConfigurationError(
                "Live mode requires a data store path, not pre-loaded RawData. "
                "Use .data(store='path/to/store.duckdb', pairs=[...]) instead."
            )

        if not self._named_data:
            raise ConfigurationError("Live mode requires a data source configured via .data(store=...)")

        first_source = next(iter(self._named_data.values()))
        if isinstance(first_source, RawData):
            raise ConfigurationError("Live mode requires a data store path, not pre-loaded RawData.")

        params = first_source
        store_path = params.get("store")
        pairs = params.get("pairs") or []
        timeframe = params.get("timeframe", "1m")

        if not store_path:
            raise ConfigurationError("Live mode requires store= path in .data()")

        from signalflow.data import StoreFactory

        raw_store = StoreFactory.create_raw_store(
            backend="duckdb",
            data_type="spot",
            db_path=store_path,
        )
        return raw_store, pairs, timeframe

    def _build_strategy_rules(self) -> tuple[list, list]:
        """Build entry/exit rule instances from config."""
        entry_rules: list = []
        exit_rules: list = []

        if self._entry_config:
            from signalflow.strategy.component.entry.signal import SignalEntryRule

            max_pos = self._entry_config.get("max_positions") or 10
            size = self._entry_config.get("size")
            if size is None:
                size = 100.0
            rule_kwargs: dict = {
                "base_position_size": float(size),
                "max_total_positions": max_pos,
            }
            if self._entry_config.get("entry_filters"):
                rule_kwargs["entry_filters"] = self._entry_config["entry_filters"]
            entry_rules.append(SignalEntryRule(**rule_kwargs))

        if self._exit_config:
            tp = self._exit_config.get("tp")
            sl = self._exit_config.get("sl")
            trailing = self._exit_config.get("trailing")

            if tp is not None or sl is not None:
                from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit

                exit_rules.append(
                    TakeProfitStopLossExit(
                        take_profit_pct=float(tp) if tp else 0.02,
                        stop_loss_pct=float(sl) if sl else 0.01,
                    )
                )

            if trailing is not None:
                from signalflow.strategy.component.exit.trailing_stop import TrailingStopExit

                exit_rules.append(TrailingStopExit(trail_pct=float(trailing)))

            # Pre-built exit rule instances (from graph converter)
            extra = self._exit_config.get("_extra_rules")
            if extra:
                exit_rules.extend(extra)

        # Defaults
        if not entry_rules:
            from signalflow.strategy.component.entry.signal import SignalEntryRule

            entry_rules.append(SignalEntryRule())
        if not exit_rules:
            from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit

            exit_rules.append(TakeProfitStopLossExit())

        return entry_rules, exit_rules

    def _resolve_strategy_store(self) -> Any:
        """Get or create a strategy store for state persistence."""
        from signalflow.data.strategy_store.memory import InMemoryStrategyStore

        return InMemoryStrategyStore()

    # =========================================================================
    # Window / Period Helpers
    # =========================================================================

    @staticmethod
    def _parse_period(period: str | int | None) -> timedelta:
        """Convert period string or int to timedelta.

        Supports: "6M" (months), "30d" (days), "1Y" (years), int (days).
        """
        if period is None:
            raise ConfigurationError("Period must be specified")
        if isinstance(period, int):
            return timedelta(days=period)

        period = period.strip()
        if period.endswith("M"):
            return timedelta(days=int(period[:-1]) * 30)
        elif period.endswith("Y"):
            return timedelta(days=int(period[:-1]) * 365)
        elif period.endswith("d"):
            return timedelta(days=int(period[:-1]))
        else:
            return timedelta(days=int(period))

    @staticmethod
    def _filter_raw_data(raw: RawData, start: datetime, end: datetime) -> RawData:
        """Create a new RawData with DataFrames filtered to [start, end)."""
        filtered_data: dict[str, Any] = {}
        for key, value in raw.data.items():
            if isinstance(value, dict):
                # Nested structure (multi-source)
                filtered_data[key] = {
                    src: df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
                    if isinstance(df, pl.DataFrame) and "timestamp" in df.columns
                    else df
                    for src, df in value.items()
                }
            elif isinstance(value, pl.DataFrame) and "timestamp" in value.columns:
                filtered_data[key] = value.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
            else:
                filtered_data[key] = value
        return RawData(
            datetime_start=start,
            datetime_end=end,
            pairs=raw.pairs,
            data=filtered_data,
            default_source=raw.default_source,
        )

    def _run_on_window(
        self,
        raw: RawData,
        train_start: datetime | None,
        train_end: datetime | None,
        test_start: datetime,
        test_end: datetime,
        *,
        current_capital: float | None = None,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
        cancel_event: Event | None = None,
    ) -> FlowResult:
        """Execute the full pipeline on a single train/test window."""
        result = FlowResult()

        # Override capital if compounding across windows
        original_capital = self._capital
        if current_capital is not None:
            self._capital = current_capital

        try:
            # Filter data to test period
            test_raw = self._filter_raw_data(raw, test_start, test_end)

            # Compute features on test data
            test_features = None
            if self._feature_pipelines:
                test_features = self._compute_features(test_raw)
                result.features = test_features

            # Detect signals on test data
            test_signals = self._resolve_signals(test_raw)
            result.signals = test_signals

            # Train validator on train data and validate test signals
            if train_start is not None and train_end is not None and self._named_validators and self._named_labelers:
                train_raw = self._filter_raw_data(raw, train_start, train_end)

                # Compute features and labels on train data
                train_features = self._compute_features(train_raw) if self._feature_pipelines else None
                train_signals = self._resolve_signals(train_raw)
                train_labels = self._compute_labels(train_raw, train_signals) if self._named_labelers else None

                if train_features is not None and train_labels is not None and train_features.height > 0:
                    # Train and validate per validator
                    for _name, validator in self._named_validators.items():
                        train_df = train_features.join(
                            train_labels.select(["pair", "timestamp", "label"]),
                            on=["pair", "timestamp"],
                            how="inner",
                        ).drop_nulls()

                        if train_df.height == 0:
                            continue

                        feature_cols = [c for c in train_df.columns if c not in ("timestamp", "pair", "label")]
                        X_train = train_df.select(feature_cols)
                        y_train = train_df.select("label")

                        # Deep copy to avoid state leakage between windows
                        fold_validator = copy.deepcopy(validator)
                        fold_validator.fit(X_train, y_train)

                        # Validate test signals
                        if test_features is not None:
                            try:
                                validated = fold_validator.predict(test_signals, test_features)
                                result.predictions = validated.value if hasattr(validated, "value") else None
                            except Exception:
                                pass

            # Apply labeling on test data (for metrics)
            if self._named_labelers:
                result.labels = self._compute_labels(test_raw, test_signals)

            # Run backtest on test data
            if self._entry_config or self._exit_config or not self._metric_nodes:
                backtest_result = self._run_backtest(test_raw, test_signals, progress_callback, cancel_event)
                if backtest_result:
                    result.trades = backtest_result.trades_df
                    result.backtest_metrics = backtest_result.metrics

            # Compute metric nodes
            self._compute_metrics(result)
        finally:
            self._capital = original_capital

        return result

    @staticmethod
    def _aggregate_results(sub_results: list[FlowResult]) -> dict[str, float]:
        """Aggregate metrics across folds/windows."""
        import numpy as np

        metrics_list = [
            r.backtest_metrics for r in sub_results if r.backtest_metrics and isinstance(r.backtest_metrics, dict)
        ]
        if not metrics_list:
            return {}

        sharpes = [m.get("sharpe_ratio", 0) for m in metrics_list]
        returns = [m.get("total_return", 0) for m in metrics_list]
        win_rates = [m.get("win_rate", 0) for m in metrics_list]
        n_trades = sum(m.get("n_trades", 0) for m in metrics_list)

        return {
            "n_folds": len(sub_results),
            "sharpe_avg": float(np.mean(sharpes)) if sharpes else 0,
            "sharpe_std": float(np.std(sharpes)) if sharpes else 0,
            "return_avg": float(np.mean(returns)) if returns else 0,
            "return_std": float(np.std(returns)) if returns else 0,
            "win_rate_avg": float(np.mean(win_rates)) if win_rates else 0,
            "n_trades": n_trades,
        }

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

        _first_name, first_source = next(iter(self._named_data.items()))

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
            "n_features": len(
                [
                    c
                    for c in features.columns
                    if c not in ("timestamp", "pair", "close", "open", "high", "low", "volume")
                ]
            ),
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
@sf_component(name="flow_feature_metrics")
class FeatureMetrics:
    """Metric node for feature analysis: correlation, importance, distribution."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    include_correlation: bool = True
    include_importance: bool = True


@dataclass
@sf_component(name="flow_signal_metrics")
class SignalMetrics:
    """Metric node for signal analysis: frequency, clustering, timing."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    include_frequency: bool = True
    include_clustering: bool = True


@dataclass
@sf_component(name="flow_label_metrics")
class LabelMetrics:
    """Metric node for label analysis: win rate distribution, holding time."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    include_distribution: bool = True
    include_holding_time: bool = True


@dataclass
@sf_component(name="flow_validation_metrics")
class ValidationMetrics:
    """Metric node for validation analysis: confusion matrix, feature importance."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    include_confusion_matrix: bool = True
    include_feature_importance: bool = True


@dataclass
@sf_component(name="flow_backtest_metrics")
class BacktestMetrics:
    """Metric node for full backtest analysis: equity curve, drawdown, Sharpe."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    include_equity_curve: bool = True
    include_drawdown: bool = True


@dataclass
@sf_component(name="flow_live_metrics")
class LiveMetrics:
    """Metric node for live trading analysis: latency, fill rate, slippage."""

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_METRIC
    include_latency: bool = True
    include_slippage: bool = True


__all__ = [
    "AggregationMode",
    "BacktestMetrics",
    # Metric nodes
    "FeatureMetrics",
    "FlowBuilder",
    "FlowConfig",
    "FlowResult",
    "LabelMetrics",
    "LiveMetrics",
    "RunMode",
    "SignalMetrics",
    "ValidationMetrics",
    "flow",
]
