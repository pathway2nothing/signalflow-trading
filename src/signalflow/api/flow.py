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
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
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
    strategy_metric,
)
from signalflow.models import ModelRef

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.signal_feature.base import SignalFeature
    from signalflow.target.base import Labeler
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


_MAX_PRICE_POINTS_PER_PAIR = 500
_MAX_EQUITY_POINTS = 2000
_MAX_FEATURE_POINTS_PER_PAIR = 2000
_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume", "trades"}


def _json_safe_value(v: Any) -> Any:
    """Convert a single value to JSON-safe type."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _dataframe_to_json_safe(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert Polars DataFrame to JSON-safe list of dicts."""
    rows = df.to_dicts()
    return [{k: _json_safe_value(v) for k, v in row.items()} for row in rows]


def _extract_price_data(raw: RawData) -> list[dict[str, Any]]:
    """Extract downsampled OHLC prices from RawData for price charting.

    Handles both flat (data_type → DataFrame) and nested
    (data_type → source → DataFrame) RawData structures.
    When downsampling, properly aggregates OHLC bars rather than
    just picking every Nth candle.
    """
    result: list[dict[str, Any]] = []

    # Collect all DataFrames from raw.data (may be flat or nested)
    dataframes: list[pl.DataFrame] = []
    for _key, value in raw.data.items():
        if value is None:
            continue
        if isinstance(value, pl.DataFrame):
            dataframes.append(value)
        elif isinstance(value, dict):
            for _src, df in value.items():
                if isinstance(df, pl.DataFrame):
                    dataframes.append(df)

    for df in dataframes:
        if df.height == 0:
            continue
        required = {"timestamp", "close", "pair"}
        if not required.issubset(set(df.columns)):
            continue

        has_ohlc = {"open", "high", "low", "close"}.issubset(set(df.columns))

        for pair in df.select("pair").unique().to_series().to_list():
            pair_df = df.filter(pl.col("pair") == pair).sort("timestamp")
            n = pair_df.height

            if n <= _MAX_PRICE_POINTS_PER_PAIR:
                sampled = pair_df
            else:
                # Properly aggregate OHLC when downsampling
                group_size = max(1, n // _MAX_PRICE_POINTS_PER_PAIR)
                idx = pl.Series("_grp", [i // group_size for i in range(n)])
                grouped = pair_df.with_columns(idx)

                agg_exprs = [
                    pl.col("pair").first(),
                    pl.col("timestamp").first(),
                    pl.col("close").last(),
                ]
                if has_ohlc:
                    agg_exprs.extend(
                        [
                            pl.col("open").first(),
                            pl.col("high").max(),
                            pl.col("low").min(),
                        ]
                    )

                sampled = grouped.group_by("_grp", maintain_order=True).agg(agg_exprs).drop("_grp")

            cols = ["pair", "timestamp"]
            for c in ["open", "high", "low", "close"]:
                if c in sampled.columns:
                    cols.append(c)
            result.extend(_dataframe_to_json_safe(sampled.select(cols)))

    return result


@dataclass
class FlowConfig:
    """Configuration snapshot of a Flow."""

    strategy_id: str
    run_mode: RunMode
    data_sources: list[str]
    feature_pipelines: list[str]  # deprecated (v2): always empty — features live in artefacts
    detectors: list[str]
    forecasts: list[str] = field(default_factory=list)
    signal_features: list[str] = field(default_factory=list)
    labelers: list[str] = field(default_factory=list)
    validators: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    capital: float = 10_000.0
    fee: float = 0.001


def pair_trades_by_position(
    trades: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pair entry/exit trades by ``position_id``.

    Returns:
        Dict of ``{position_id: {"entries": [...], "exit": {...} | None,
        "pnl": float | None}}``.
    """
    entries_by_pos: dict[str, list[dict[str, Any]]] = {}
    exits_by_pos: dict[str, dict[str, Any]] = {}

    for t in trades:
        meta = t.get("meta") or {}
        pos_id = t.get("position_id")
        if not pos_id:
            continue
        if isinstance(meta, dict) and meta.get("type") == "entry":
            entries_by_pos.setdefault(pos_id, []).append(t)
        elif isinstance(meta, dict) and meta.get("type") == "exit":
            exits_by_pos[pos_id] = t

    result: dict[str, dict[str, Any]] = {}
    all_pos_ids = set(entries_by_pos.keys()) | set(exits_by_pos.keys())
    for pos_id in all_pos_ids:
        entries = entries_by_pos.get(pos_id, [])
        exit_trade = exits_by_pos.get(pos_id)
        pnl: float | None = None
        if entries and exit_trade:
            entry_notional = sum(e["price"] * e["qty"] for e in entries)
            entry_fees = sum(e.get("fee", 0) for e in entries)
            exit_notional = exit_trade["price"] * exit_trade["qty"]
            exit_fee = exit_trade.get("fee", 0)
            if entries[0].get("side") == "BUY":
                pnl = exit_notional - entry_notional - entry_fees - exit_fee
            else:
                pnl = entry_notional - exit_notional - entry_fees - exit_fee
        result[pos_id] = {"entries": entries, "exit": exit_trade, "pnl": pnl}
    return result


def compute_trade_pnl(entries: list[dict[str, Any]], exit_trade: dict[str, Any]) -> float:
    """Compute PnL for a single closed position.

    Args:
        entries: List of entry trade dicts for this position.
        exit_trade: The exit trade dict.

    Returns:
        PnL in quote currency (positive = profit).
    """
    entry_notional = sum(e["price"] * e["qty"] for e in entries)
    entry_fees = sum(e.get("fee", 0) for e in entries)
    exit_notional = exit_trade["price"] * exit_trade["qty"]
    exit_fee = exit_trade.get("fee", 0)
    if entries[0].get("side") == "BUY":
        return float(exit_notional - entry_notional - entry_fees - exit_fee)
    return float(entry_notional - exit_notional - entry_fees - exit_fee)


def enrich_trades_with_pnl(trades: list[dict[str, Any]]) -> None:
    """Add ``pnl`` field to exit trades by pairing with entry trades via position_id.

    Mutates the trade dicts in-place.
    """
    _enrich_trades_with_pnl(trades)


def _enrich_trades_with_pnl(trades: list[dict[str, Any]]) -> None:
    """Add ``pnl`` field to exit trades by pairing with entry trades via position_id."""
    entries_by_pos: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        meta = t.get("meta") or {}
        if isinstance(meta, dict) and meta.get("type") == "entry":
            pos_id = t.get("position_id")
            if pos_id:
                entries_by_pos.setdefault(pos_id, []).append(t)

    for t in trades:
        meta = t.get("meta") or {}
        if not isinstance(meta, dict) or meta.get("type") != "exit":
            continue
        pos_id = t.get("position_id")
        entries = entries_by_pos.get(pos_id, [])  # type: ignore[arg-type]
        if not entries:
            continue
        # Sum all entry notional and fees for this position
        entry_notional = sum(e["price"] * e["qty"] for e in entries)
        entry_fees = sum(e.get("fee", 0) for e in entries)
        exit_notional = t["price"] * t["qty"]
        exit_fee = t.get("fee", 0)
        # Long (entry=BUY): pnl = exit - entry - fees
        # Short (entry=SELL): pnl = entry - exit - fees
        if entries[0].get("side") == "BUY":
            t["pnl"] = exit_notional - entry_notional - entry_fees - exit_fee
        else:
            t["pnl"] = entry_notional - exit_notional - entry_fees - exit_fee


def _lttb_indices(y: np.ndarray, target: int) -> list[int]:
    """Largest Triangle Three Buckets downsampling.

    Returns *target* indices that best preserve the visual shape of *y*.
    """
    n = len(y)
    if n <= target:
        return list(range(n))

    indices = [0]
    bucket_size = (n - 2) / (target - 2)

    a = 0  # previous selected index
    for i in range(1, target - 1):
        # Next bucket boundaries
        b_start = int((i - 1) * bucket_size) + 1
        b_end = int(i * bucket_size) + 1
        # Average of the bucket after this one (lookahead)
        c_start = int(i * bucket_size) + 1
        c_end = int((i + 1) * bucket_size) + 1
        c_end = min(c_end, n)
        avg_y = float(np.nanmean(y[c_start:c_end])) if c_end > c_start else 0.0

        # Pick the point in current bucket with largest triangle area
        best_idx = b_start
        best_area = -1.0
        for j in range(b_start, min(b_end, n)):
            area = abs((j - a) * (avg_y - y[a]) - (float(np.nan_to_num(y[j])) - y[a]) * (c_start - a))
            if area > best_area:
                best_area = area
                best_idx = j
        indices.append(best_idx)
        a = best_idx

    indices.append(n - 1)
    return indices


def _downsample_detector_features(
    features_map: dict[str, pl.DataFrame],
) -> dict[str, list[dict[str, Any]]]:
    """Downsample detector features for JSON serialization.

    Keeps only computed feature columns (not raw OHLCV), plus timestamp and pair.
    Uses LTTB (Largest Triangle Three Buckets) to preserve visual shape.
    Limits to _MAX_FEATURE_POINTS_PER_PAIR per pair.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for det_name, df in features_map.items():
        # Keep feature columns + timestamp + pair
        feature_cols = [c for c in df.columns if c not in _OHLCV_COLUMNS]
        if not {"timestamp", "pair"}.issubset(set(feature_cols)):
            continue
        if len(feature_cols) <= 2:  # only timestamp + pair, no features
            continue
        fdf = df.select(feature_cols)

        per_pair: list[dict[str, Any]] = []
        for pair in fdf.select("pair").unique().to_series().to_list():
            pair_df = fdf.filter(pl.col("pair") == pair).sort("timestamp")
            n = pair_df.height
            if n > _MAX_FEATURE_POINTS_PER_PAIR:
                # Use first numeric feature column for LTTB pivot
                num_cols = [c for c in feature_cols if c not in ("timestamp", "pair")]
                y = pair_df[num_cols[0]].fill_null(0).to_numpy()
                idx = _lttb_indices(y, _MAX_FEATURE_POINTS_PER_PAIR)
                pair_df = pair_df[idx]
            per_pair.extend(_dataframe_to_json_safe(pair_df))
        if per_pair:
            result[det_name] = per_pair
    return result


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
    signal_feature_df: pl.DataFrame | None = None
    signals: Signals | None = None
    labels: pl.DataFrame | None = None
    predictions: pl.DataFrame | None = None
    trades: pl.DataFrame | None = None
    detector_features: dict[str, pl.DataFrame] | None = None

    # Chart data (serialized for API/UI)
    equity_curve: list[dict[str, Any]] | None = None
    price_data: list[dict[str, Any]] | None = None

    # Metadata
    flow_config: FlowConfig | None = None
    execution_time: float = 0.0
    warnings: list[str] | None = None
    data_warnings: list[dict[str, Any]] | None = None
    data_start: str | None = None
    data_end: str | None = None
    strategy_summary: dict[str, Any] | None = None

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

        result["n_signals"] = self.signals.value.height if self.signals and hasattr(self.signals, "value") else 0
        result["n_trades"] = self.trades.height if self.trades is not None else 0
        result["execution_time"] = self.execution_time

        if self.trades is not None and self.trades.height > 0:
            trade_dicts = self.trades.to_dicts()
            _enrich_trades_with_pnl(trade_dicts)
            result["trades"] = trade_dicts

        if self.equity_curve:
            result["equity_curve"] = self.equity_curve

        if self.price_data:
            result["price_data"] = self.price_data

        if self.flow_config:
            result["config"] = {
                "strategy_id": self.flow_config.strategy_id,
                "run_mode": self.flow_config.run_mode.value,
                "capital": self.flow_config.capital,
            }
        if self.data_start or self.data_end:
            cfg = result.setdefault("config", {})
            if self.data_start:
                cfg["data_start"] = self.data_start
            if self.data_end:
                cfg["data_end"] = self.data_end

        if self.window_results:
            result["window_results"] = [wr.to_json_dict() for wr in self.window_results]
        if self.fold_results:
            result["fold_results"] = [fr.to_json_dict() for fr in self.fold_results]

        if self.detector_features:
            result["detector_features"] = _downsample_detector_features(self.detector_features)

        if self.warnings:
            result["warnings"] = self.warnings

        if self.data_warnings:
            result["data_warnings"] = self.data_warnings

        if self.strategy_summary:
            result["strategy_summary"] = self.strategy_summary

        return result


@dataclass
class FlowBuilder:
    """
    Fluent builder for unified SignalFlow pipelines.

    FlowBuilder composes:
    - Pinned forecast artefacts (.forecast()) — features live inside the artefact
    - Signal detection (.detector(), with optional forecasts=)
    - Target labeling (.labeler())
    - Validation / meta-labeling (.validator())
    - Metric nodes (.metrics())
    - Multiple run modes (.run(mode=...))
    - Artifact saving

    Example (full backtest):
        >>> result = (
        ...     sf.flow()
        ...     .data(store="binance", pair="BTC/USDT")
        ...     .forecast("revert", mlflow="models:/revert/3")
        ...     .detector("rsi_cross", forecasts=["revert"], forecast_window=30)
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

    # Forecasts (pinned ML artefacts — declared lazily via .forecast(); weights load at .run())
    _named_forecasts: dict[str, ModelRef] = field(default_factory=dict, repr=False)
    # Which named forecasts each consumer (detector/validator/sizing/exit) reads, and its window.
    _forecast_consumers: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _forecast_windows: dict[str, int] = field(default_factory=dict, repr=False)

    # Detection
    _named_detectors: dict[str, SignalDetector] = field(default_factory=dict, repr=False)
    _signals: Signals | None = field(default=None, repr=False)
    _aggregation_config: dict[str, Any] | None = field(default=None, repr=False)

    # Signal features (meta-features from signal history)
    _signal_features: list[SignalFeature] = field(default_factory=list, repr=False)

    # Labeling
    _named_labelers: dict[str, Labeler] = field(default_factory=dict, repr=False)

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
    # Forecast Configuration
    # =========================================================================
    #
    # NOTE (v2 refactor, VISION §4.2): the former ``.features()`` builder method and
    # ``_feature_pipelines`` were removed. ``flow`` is a composer of artefacts, not a
    # feature constructor. Features now live INSIDE a forecast artefact (pinned with its
    # weights) or as primitive parameters on a detector. Declare models via ``.forecast()``.

    def forecast(
        self,
        name: str,
        *,
        mlflow: str | None = None,
        hf_path: str | None = None,
        version: str | int | None = None,
        source: str | None = None,
    ) -> Self:
        """Register a pinned forecast-model artefact (lazy — no weights loaded here).

        The forecast is referenced by ``name`` from consumers via ``forecasts=[name]``
        on ``.detector()``/``.validator()``/``.exit()``. Only a :class:`ModelRef` is
        recorded; weights resolve through the registry at ``.run()`` (VISION §4.3).

        Args:
            name: Local name to reference this forecast by.
            mlflow: MLflow model URI, e.g. ``"models:/revert/3"`` (version inside the URI).
            hf_path: HuggingFace Hub path (``source="hf"``); pair with ``version=``.
            version: Explicit version when not embedded in the URI. ``latest`` is rejected
                unless ``SF_ALLOW_LATEST=1`` — otherwise parity/reproducibility is lost.
            source: Override artefact source (``"mlflow"`` | ``"hf"``); inferred otherwise.

        Returns:
            Self for method chaining.
        """
        if name in self._named_forecasts:
            raise ConfigurationError(f"Forecast '{name}' already registered")

        if mlflow is not None:
            ref = ModelRef.parse(mlflow, source="mlflow")
            if version is not None and str(version) != str(ref.version):
                raise ConfigurationError(
                    f"Conflicting version for forecast '{name}': URI says '{ref.version}', version= says '{version}'",
                )
        elif hf_path is not None:
            if version is None:
                raise ConfigurationError(
                    f"Forecast '{name}' from hf_path requires explicit version=",
                )
            ref = ModelRef(name=hf_path, version=version, source=source or "hf")
        else:
            raise ConfigurationError(
                f"Forecast '{name}' needs either mlflow= or hf_path=",
            )

        self._named_forecasts[name] = ref
        return self

    # =========================================================================
    # Detector Configuration
    # =========================================================================

    def detector(
        self,
        detector: SignalDetector | str,
        *,
        name: str | None = None,
        forecasts: list[str] | None = None,
        forecast_window: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a signal detector to the flow.

        Args:
            detector: SignalDetector instance or registry name
            name: Unique name for cross-referencing
            forecasts: Names of registered forecasts (see ``.forecast()``) this detector
                reads. The detector sees a WINDOW of forecast values ``[t-w, t]``, not just
                the latest point (VISION §6.2).
            forecast_window: Window length in BARS (not "however much accumulated"). Required
                when ``forecasts`` is given — fixes the warmup-silence contract so backtest and
                live cold-start cut the identical slice and parity holds.
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
        self._register_forecast_consumer(name, forecasts, forecast_window)
        return self

    def _register_forecast_consumer(
        self,
        consumer: str,
        forecasts: list[str] | None,
        forecast_window: int | None,
    ) -> None:
        """Record which forecasts a consumer reads + its window (warmup-silence contract).

        Enforces VISION §6.2: a consumer of ``forecasts=`` MUST declare a fixed bar-count
        window. Existence of the referenced forecasts is checked lazily (forecasts may be
        declared after their consumer) via :meth:`_validate_forecast_refs`.
        """
        if not forecasts:
            if forecast_window is not None:
                raise ConfigurationError(
                    f"'{consumer}': forecast_window set without forecasts=",
                )
            return
        if forecast_window is None or forecast_window <= 0:
            raise ConfigurationError(
                f"'{consumer}': forecasts= requires a positive forecast_window "
                "(window must be fixed in bars, not 'whatever accumulated' — VISION §6.2)",
            )
        self._forecast_consumers[consumer] = list(forecasts)
        self._forecast_windows[consumer] = forecast_window

    def _validate_forecast_refs(self) -> None:
        """Every forecast referenced by a consumer must be registered via .forecast()."""
        for consumer, names in self._forecast_consumers.items():
            missing = [n for n in names if n not in self._named_forecasts]
            if missing:
                raise ConfigurationError(
                    f"'{consumer}' references unknown forecast(s) {missing}; "
                    f"registered: {sorted(self._named_forecasts)}",
                )

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
    # Signal Feature Configuration
    # =========================================================================

    def signal_features(
        self,
        features: SignalFeature | list[SignalFeature],
    ) -> Self:
        """Add signal-level features (meta-features from signal history).

        Signal features compute statistics about the signal stream itself
        (frequency, entropy, rolling accuracy, etc.) and produce a
        separate feature DataFrame that is joined with raw features
        before validation.

        Args:
            features: A single SignalFeature or list of SignalFeatures.

        Returns:
            Self for method chaining.

        Example:
            >>> flow = (
            ...     sf.flow()
            ...     .data(raw)
            ...     .detector("rsi_cross")
            ...     .signal_features([
            ...         SignalFrequency(window=50),
            ...         RollingAccuracy(window=100),
            ...     ])
            ...     .validator("lightgbm")
            ...     .run()
            ... )
        """
        if isinstance(features, list):
            self._signal_features.extend(features)
        else:
            self._signal_features.append(features)
        return self

    # =========================================================================
    # Labeler Configuration
    # =========================================================================

    def labeler(
        self,
        labeler: Labeler | str,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a target labeler to the flow.

        Args:
            labeler: Labeler instance or registry name
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
        forecasts: list[str] | None = None,
        forecast_window: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a signal validator (meta-labeler) to the flow.

        Args:
            validator: SignalValidator instance or registry name
            name: Unique name for cross-referencing
            forecasts: Names of registered forecasts this validator reads (windowed, §6.2).
            forecast_window: Window length in bars; required when ``forecasts`` is given.
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
        self._register_forecast_consumer(f"validator:{name}", forecasts, forecast_window)
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
        forecasts: list[str] | None = None,
        forecast_window: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Configure exit rules.

        ``forecasts`` / ``forecast_window`` let the exit (ExitPolicy) read a windowed
        forecast slice (§6.2); ``forecast_window`` is required when ``forecasts`` is set.
        """
        self._exit_config = {
            "rule": rule,
            "tp": tp,
            "sl": sl,
            "trailing": trailing,
            **kwargs,
        }
        self._register_forecast_consumer("exit", forecasts, forecast_window)
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

        # Forecast references must resolve to declared .forecast() artefacts.
        self._validate_forecast_refs()

        # Build config snapshot
        config = FlowConfig(
            strategy_id=self.strategy_id,
            run_mode=run_mode,
            data_sources=list(self._named_data.keys()),
            feature_pipelines=[],  # deprecated (v2): features live in forecast artefacts
            detectors=list(self._named_detectors.keys()),
            forecasts=list(self._named_forecasts.keys()),
            signal_features=[type(sf).__name__ for sf in self._signal_features],
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

        from loguru import logger as _logger

        # 1. Resolve data
        raw = self._resolve_data()
        _logger.info(
            "Data loaded: {} pairs, {} to {}",
            len(raw.pairs) if hasattr(raw, "pairs") else "?",
            getattr(raw, "datetime_start", "?"),
            getattr(raw, "datetime_end", "?"),
        )

        # Extract actual data period from RawData
        if hasattr(raw, "datetime_start") and raw.datetime_start:
            ds = raw.datetime_start
            result.data_start = ds.isoformat() if hasattr(ds, "isoformat") else str(ds)
        if hasattr(raw, "datetime_end") and raw.datetime_end:
            de = raw.datetime_end
            result.data_end = de.isoformat() if hasattr(de, "isoformat") else str(de)

        # 2. Detect signals (capturing detector-preprocessed features)
        signals, det_features = self._resolve_signals(raw)
        result.signals = signals

        # v2 (VISION §6.1): the feature matrix for validation comes from detector
        # preprocessing / forecast artefacts — not from a separate .features() pipeline.
        features_df = self._merge_detector_features(det_features)
        if features_df is not None:
            result.features = features_df
        _logger.info(
            "Signal detection: {} signals from {} detector(s)",
            signals.value.height if signals else 0,
            len(self._named_detectors),
        )
        if det_features:
            result.detector_features = det_features

        # 4. Apply labeling (if configured)
        labels_df = None
        if self._named_labelers:
            labels_df = self._compute_labels(raw, signals)
            result.labels = labels_df

        # 4b. Compute signal features (meta-features from signal history)
        if self._signal_features and signals is not None:
            sig_feat_df = self._compute_signal_features(signals, labels_df)
            result.signal_feature_df = sig_feat_df
            _logger.info(
                "Signal features: {} feature(s), {} columns",
                len(self._signal_features),
                sig_feat_df.width - 2 if sig_feat_df is not None else 0,
            )
            # Merge signal features into the main feature matrix
            if features_df is not None and sig_feat_df is not None:
                features_df = features_df.join(sig_feat_df, on=["pair", "timestamp"], how="left")
                result.features = features_df
            elif sig_feat_df is not None:
                features_df = sig_feat_df
                result.features = features_df

        # 5. Apply validation (if configured)
        if self._named_validators and labels_df is not None and features_df is not None:
            result.predictions = self._apply_validation(signals, features_df, labels_df)

        # 6. Run backtest (if strategy configured)
        if self._entry_config or self._exit_config or not self._metric_nodes:
            _logger.info("Running backtest...")
            backtest_result = self._run_backtest(raw, signals, progress_callback, cancel_event)
            if backtest_result:
                result.trades = backtest_result.trades_df
                result.backtest_metrics = backtest_result.metrics
                n_trades = len(backtest_result.trades) if hasattr(backtest_result, "trades") else 0
                _logger.info(
                    "Backtest complete: {} trades, final capital ${:,.0f}",
                    n_trades,
                    backtest_result.metrics.get("final_capital", 0) if backtest_result.metrics else 0,
                )

                # Extract equity curve from metrics_df (downsampled for large backtests)
                if hasattr(backtest_result, "metrics_df") and backtest_result.metrics_df is not None:
                    mdf = backtest_result.metrics_df
                    if mdf.height > 0:
                        if mdf.height > _MAX_EQUITY_POINTS:
                            step = max(1, mdf.height // _MAX_EQUITY_POINTS)
                            indices = list(range(0, mdf.height, step))
                            if indices[-1] != mdf.height - 1:
                                indices.append(mdf.height - 1)
                            mdf = mdf[indices]
                        result.equity_curve = _dataframe_to_json_safe(mdf)

                # Extract downsampled close prices from raw data
                result.price_data = _extract_price_data(raw)

        # 6b. Capture strategy configuration summary
        result.strategy_summary = self._build_strategy_summary()

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

        from signalflow.strategy.broker.executor.base import OrderExecutor
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
        broker = VirtualRealtimeBroker(executor=cast(OrderExecutor, executor), store=store)

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

        from signalflow.core.containers.portfolio import Portfolio

        trades_list = runner.trades
        trades_df = Portfolio.trades_to_pl(trades_list) if trades_list else pl.DataFrame()
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
            size_pct = self._entry_config.get("size_pct")
            if size_pct:
                size = self._capital * size_pct
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

            # Detect signals on test data (features come from detector preprocessing, v2)
            test_signals, det_features = self._resolve_signals(test_raw)
            result.signals = test_signals
            if det_features:
                result.detector_features = det_features
            test_features = self._merge_detector_features(det_features)
            if test_features is not None:
                result.features = test_features

            # Train validator on train data and validate test signals
            if train_start is not None and train_end is not None and self._named_validators and self._named_labelers:
                train_raw = self._filter_raw_data(raw, train_start, train_end)

                # Features and labels on train data (features from detector preprocessing)
                train_signals, train_det_features = self._resolve_signals(train_raw)
                train_features = self._merge_detector_features(train_det_features)
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
    # Public Pipeline Stage Access
    # =========================================================================

    def resolve_data(self) -> RawData:
        """Resolve and load data sources.

        Returns:
            Loaded RawData from configured data sources.

        Raises:
            MissingDataError: If no data source is configured.
        """
        return self._resolve_data()

    def compute_features(self, raw: RawData | None = None) -> pl.DataFrame:
        """Return the feature matrix produced by detector preprocessing (v2).

        ``flow`` no longer owns feature pipelines (VISION §4.2); features come from the
        detectors' own preprocessing / forecast artefacts. Requires at least one detector.

        Args:
            raw: Pre-loaded RawData. If None, resolves from config.

        Raises:
            MissingDataError: If no data and raw is None.
            ConfigurationError: If no detector is configured.
        """
        if raw is None:
            raw = self._resolve_data()
        if not self._named_detectors:
            raise ConfigurationError("No detector configured to produce features")
        _signals, det_features = self._resolve_signals(raw)
        merged = self._merge_detector_features(det_features)
        if merged is None:
            raise ConfigurationError("Detectors produced no features")
        return merged

    def resolve_signals(self, raw: RawData | None = None) -> tuple[Signals, dict[str, pl.DataFrame]]:
        """Detect signals from all detectors, capturing preprocessed features.

        Args:
            raw: Pre-loaded RawData. If None, resolves from config.

        Returns:
            Tuple of (merged signals, detector feature map).
        """
        if raw is None:
            raw = self._resolve_data()
        return self._resolve_signals(raw)

    def compute_labels(
        self,
        raw: RawData | None = None,
        signals: Signals | None = None,
    ) -> pl.DataFrame:
        """Compute labels from all configured labelers.

        Args:
            raw: Pre-loaded RawData. If None, resolves from config.
            signals: Pre-computed signals. If None, resolves from detectors.

        Returns:
            DataFrame with labels.

        Raises:
            ConfigurationError: If no labelers configured.
        """
        if raw is None:
            raw = self._resolve_data()
        if signals is None:
            signals, _ = self._resolve_signals(raw)
        if not self._named_labelers:
            raise ConfigurationError("No labelers configured")
        return self._compute_labels(raw, signals)

    def build_rules(self) -> tuple[list, list]:
        """Build entry/exit rule instances from the current configuration.

        Returns:
            Tuple of (entry_rules, exit_rules).
        """
        return self._build_strategy_rules()

    # =========================================================================
    # Public Config Introspection & Mutation
    # =========================================================================

    def update_config(
        self,
        *,
        detector: dict[str, Any] | None = None,
        entry: dict[str, Any] | None = None,
        exit: dict[str, Any] | None = None,
    ) -> Self:
        """Update component parameters after initial configuration.

        Useful for hyperparameter tuning — mutate params without
        rebuilding the entire builder.

        Args:
            detector: Dict of {param_name: value} applied to all detectors.
            entry: Dict of {param_name: value} merged into entry config.
            exit: Dict of {param_name: value} merged into exit config.

        Returns:
            Self for method chaining.
        """
        if detector:
            for _name, det in self._named_detectors.items():
                for k, v in detector.items():
                    if hasattr(det, k):
                        setattr(det, k, v)

        if entry and self._entry_config:
            for k, v in entry.items():
                if k in self._entry_config:
                    self._entry_config[k] = v

        if exit and self._exit_config:
            for k, v in exit.items():
                if k in self._exit_config:
                    self._exit_config[k] = v

        return self

    def get_tunable_params(self) -> dict[str, dict[str, Any]]:
        """Extract tunable parameters with their current values and types.

        Returns:
            Dict of ``{param_name: {"value": current, "type": "int"|"float",
            "source": "detector"|"entry"|"exit"}}``.
        """
        import dataclasses as _dc

        params: dict[str, dict[str, Any]] = {}

        # Detector params
        for _name, det in self._named_detectors.items():
            if _dc.is_dataclass(det):
                for f in _dc.fields(det):
                    if f.name.startswith("_") or f.name == "component_type":
                        continue
                    val = getattr(det, f.name, None)
                    if isinstance(val, int) and val > 0:
                        params[f.name] = {
                            "value": val,
                            "type": "int",
                            "source": "detector",
                            "low": max(1, val // 2),
                            "high": val * 2,
                        }
                    elif isinstance(val, float) and val > 0:
                        params[f.name] = {
                            "value": val,
                            "type": "float",
                            "source": "detector",
                            "low": val * 0.5,
                            "high": val * 2.0,
                        }

        # Entry params
        for k, v in self._entry_config.items():
            if k.startswith("_"):
                continue
            if isinstance(v, int) and v > 0:
                params[k] = {"value": v, "type": "int", "source": "entry", "low": max(1, v // 2), "high": v * 2}
            elif isinstance(v, float) and v > 0:
                params[k] = {"value": v, "type": "float", "source": "entry", "low": v * 0.5, "high": v * 2.0}

        # Exit params
        for k, v in self._exit_config.items():
            if k.startswith("_"):
                continue
            if isinstance(v, int) and v > 0:
                params[k] = {"value": v, "type": "int", "source": "exit", "low": max(1, v // 2), "high": v * 2}
            elif isinstance(v, float) and v > 0:
                params[k] = {"value": v, "type": "float", "source": "exit", "low": v * 0.5, "high": v * 2.0}

        return params

    def clone(
        self,
        *,
        strategy_id: str | None = None,
        detector: dict[str, Any] | None = None,
        entry: dict[str, Any] | None = None,
        exit: dict[str, Any] | None = None,
        capital: float | None = None,
        fee: float | None = None,
    ) -> FlowBuilder:
        """Create a deep copy with optional parameter overrides.

        Unlike :meth:`update_config` which mutates in-place, ``clone()``
        returns a new independent FlowBuilder.

        Args:
            strategy_id: Override strategy ID.
            detector: Dict of {param_name: value} to override on detectors.
            entry: Dict of {param_name: value} to merge into entry config.
            exit: Dict of {param_name: value} to merge into exit config.
            capital: Override capital amount.
            fee: Override fee rate.

        Returns:
            New FlowBuilder with overridden parameters.
        """
        new = copy.deepcopy(self)
        if strategy_id is not None:
            new.strategy_id = strategy_id
        if capital is not None:
            new._capital = capital
        if fee is not None:
            new._fee = fee
        if detector or entry or exit:
            new.update_config(detector=detector, entry=entry, exit=exit)
        return new

    def sweep(
        self,
        param_grid: dict[str, list[Any]],
        *,
        parallel: bool = False,
        max_workers: int = 4,
        run_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Run a parameter sweep over a grid of values.

        Creates one FlowBuilder clone per combination of parameters
        and runs them all via :func:`batch_run`.

        Args:
            param_grid: Dict mapping parameter paths to lists of values.
                Paths use dot notation: ``"detector.fast_period"``,
                ``"entry.size_pct"``, ``"exit.tp"``, ``"capital"``, ``"fee"``.
            parallel: Run configs in parallel threads.
            max_workers: Thread count for parallel mode.
            run_kwargs: Extra kwargs for ``FlowBuilder.run()``.

        Returns:
            BatchResult with one result per parameter combination.

        Example:
            >>> results = base.sweep({
            ...     "detector.fast_period": [10, 20, 30],
            ...     "exit.tp": [0.02, 0.03],
            ... })
            >>> print(results.comparison.summary())
        """
        import itertools

        from signalflow.api.batch import batch_run

        # Parse param grid into (path, values) tuples
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        configs: list[FlowBuilder] = []
        labels: list[str] = []

        for combo in itertools.product(*value_lists):
            overrides: dict[str, dict[str, Any]] = {
                "detector": {},
                "entry": {},
                "exit": {},
            }
            sid_parts: list[str] = []
            clone_kwargs: dict[str, Any] = {}

            for key, val in zip(keys, combo, strict=True):
                sid_parts.append(f"{key.split('.')[-1]}={val}")

                if "." in key:
                    category, param = key.split(".", 1)
                    if category in overrides:
                        overrides[category][param] = val
                    elif category in ("capital", "fee"):
                        clone_kwargs[category] = val
                elif key in ("capital", "fee"):
                    clone_kwargs[key] = val
                else:
                    # Default: treat as detector param
                    overrides["detector"][key] = val

            clone_kwargs["strategy_id"] = ",".join(sid_parts)
            # Only pass non-empty dicts
            if overrides["detector"]:
                clone_kwargs["detector"] = overrides["detector"]
            if overrides["entry"]:
                clone_kwargs["entry"] = overrides["entry"]
            if overrides["exit"]:
                clone_kwargs["exit"] = overrides["exit"]

            configs.append(self.clone(**clone_kwargs))
            labels.append(",".join(sid_parts))

        return batch_run(
            configs,
            labels=labels,
            parallel=parallel,
            max_workers=max_workers,
            run_kwargs=run_kwargs,
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _resolve_data(self) -> RawData:
        """Resolve data sources."""
        from loguru import logger as _logger

        if self._raw is not None:
            _logger.debug("Using pre-loaded RawData")
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
            _logger.debug(
                "Loading data source '{}': {}",
                _first_name,
                {k: v for k, v in params.items() if k != "pairs"},
            )
            return load(**params)

        raise MissingDataError()

    def _merge_detector_features(
        self,
        det_features: dict[str, pl.DataFrame],
    ) -> pl.DataFrame | None:
        """Build the feature matrix for validation from detector-preprocessed features.

        v2 (VISION §6.1): features are produced by detector preprocessing (primitive params)
        or forecast artefacts — there is no separate ``.features()`` pipeline. For a single
        detector this is its feature frame; for several, outer-join on ``(pair, timestamp)``.
        """
        if not det_features:
            return None
        frames = list(det_features.values())
        merged = frames[0]
        for nxt in frames[1:]:
            new_cols = [c for c in nxt.columns if c not in ("pair", "timestamp")]
            merged = merged.join(
                nxt.select(["pair", "timestamp", *new_cols]),
                on=["pair", "timestamp"],
                how="full",
                coalesce=True,
            )
        return merged

    def _resolve_signals(self, raw: RawData) -> tuple[Signals, dict[str, pl.DataFrame]]:
        """Detect signals from all detectors, capturing preprocessed features."""
        from loguru import logger as _logger

        detector_features_map: dict[str, pl.DataFrame] = {}

        if self._signals is not None:
            return self._signals, detector_features_map

        if not self._named_detectors:
            empty = Signals(pl.DataFrame(schema={"timestamp": pl.Datetime, "pair": pl.Utf8, "signal": pl.Int8}))
            return empty, detector_features_map

        from signalflow.core.containers.raw_data_view import RawDataView

        view = RawDataView(raw=raw)
        signals_list = []

        for name, detector in self._named_detectors.items():
            # Inline the detector pipeline to capture preprocessed features
            feats = detector.preprocess(view, context=None)
            feats = detector._normalize_index(feats)
            detector._validate_features(feats)

            signals = detector.detect(feats, context=None)
            detector._validate_signals(signals)

            if detector.keep_only_latest_per_pair:
                signals = detector._keep_only_latest(signals)

            _logger.debug(
                "Detector '{}': {} features → {} signals",
                name,
                feats.height,
                signals.value.height,
            )
            signals_list.append(signals)
            detector_features_map[name] = feats

        if len(signals_list) == 1:
            return signals_list[0], detector_features_map

        from functools import reduce

        return reduce(lambda a, b: a + b, signals_list), detector_features_map

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
            return cast(pl.DataFrame, results[0])

        # Merge labels
        return cast(pl.DataFrame, pl.concat(results).unique(["pair", "timestamp"]))

    def _compute_signal_features(
        self,
        signals: Signals,
        labels_df: pl.DataFrame | None,
    ) -> pl.DataFrame | None:
        """Compute signal-level meta-features.

        Each :class:`SignalFeature` produces a DataFrame keyed on
        ``(pair, timestamp)``.  Results are joined horizontally.
        """
        from loguru import logger as _logger

        key = ["pair", "timestamp"]
        result: pl.DataFrame | None = None

        for sf_feat in self._signal_features:
            # Supervised features need labels
            if sf_feat.requires_labels and labels_df is None:
                _logger.warning(
                    "Skipping {} (requires labels but none available)",
                    type(sf_feat).__name__,
                )
                continue

            feat_df = sf_feat(
                signals=signals.value,
                labels=labels_df,
            )

            if result is None:
                result = feat_df
            else:
                new_cols = [c for c in feat_df.columns if c not in key]
                result = result.join(feat_df.select([*key, *new_cols]), on=key, how="left")

        return result

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
        # Propagate named detectors so _resolve_signals returns proper named_signals
        # (pre-computed _signals returns empty dict, breaking source_detector lookup)
        builder._named_detectors = dict(self._named_detectors)
        builder._signals = signals
        builder._entry_config = self._entry_config
        builder._exit_config = self._exit_config
        builder._capital = self._capital
        builder._fee = self._fee

        return builder.run(progress_callback=progress_callback, cancel_event=cancel_event)

    def _build_strategy_summary(self) -> dict[str, Any]:
        """Build a JSON-safe summary of the strategy configuration."""
        summary: dict[str, Any] = {
            "capital": self._capital,
            "fee": self._fee,
        }

        # Detectors
        if self._named_detectors:
            summary["detectors"] = list(self._named_detectors.keys())

        # Forecasts (pinned artefacts)
        if self._named_forecasts:
            summary["forecasts"] = list(self._named_forecasts.keys())

        # Entry config (safe keys only)
        if self._entry_config:
            entry = {}
            for k, v in self._entry_config.items():
                if k.startswith("_"):
                    continue
                is_scalar = isinstance(v, (str, int, float, bool, type(None)))
                is_str_list = isinstance(v, list) and all(isinstance(x, str) for x in v)
                if is_scalar or is_str_list:
                    entry[k] = v
            if entry:
                summary["entry"] = entry

        # Exit config (safe keys, serialize rule instances)
        if self._exit_config:
            exit_info: dict[str, Any] = {}
            for k, v in self._exit_config.items():
                if k == "_extra_rules":
                    exit_info["custom_rules"] = [
                        {
                            "type": type(r).__name__,
                            **{
                                f.name: getattr(r, f.name)
                                for f in r.__dataclass_fields__.values()
                                if f.name != "component_type"
                                and isinstance(getattr(r, f.name, None), (str, int, float, bool, type(None)))
                            },
                        }
                        for r in v
                        if hasattr(r, "__dataclass_fields__")
                    ]
                elif isinstance(v, (str, int, float, bool, type(None))):
                    exit_info[k] = v
            if exit_info:
                summary["exit"] = exit_info

        return summary

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
        if self._named_forecasts:
            parts.append(f"forecasts={list(self._named_forecasts.keys())}")
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
@strategy_metric("flow_feature_metrics")
class FeatureMetrics:
    """Metric node for feature analysis: correlation, importance, distribution."""

    include_correlation: bool = True
    include_importance: bool = True


@dataclass
@strategy_metric("flow_signal_metrics")
class SignalMetrics:
    """Metric node for signal analysis: frequency, clustering, timing."""

    include_frequency: bool = True
    include_clustering: bool = True


@dataclass
@strategy_metric("flow_label_metrics")
class LabelMetrics:
    """Metric node for label analysis: win rate distribution, holding time."""

    include_distribution: bool = True
    include_holding_time: bool = True


@dataclass
@strategy_metric("flow_validation_metrics")
class ValidationMetrics:
    """Metric node for validation analysis: confusion matrix, feature importance."""

    include_confusion_matrix: bool = True
    include_feature_importance: bool = True


@dataclass
@strategy_metric("flow_backtest_metrics")
class BacktestMetrics:
    """Metric node for full backtest analysis: equity curve, drawdown, Sharpe."""

    include_equity_curve: bool = True
    include_drawdown: bool = True


@dataclass
@strategy_metric("flow_live_metrics")
class LiveMetrics:
    """Metric node for live trading analysis: latency, fill rate, slippage."""

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
    # Public utilities
    "compute_trade_pnl",
    "enrich_trades_with_pnl",
    "flow",
    "pair_trades_by_position",
]
