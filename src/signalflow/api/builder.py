"""
BacktestBuilder - Fluent builder for backtest configuration.

Uses SignalFlowRegistry for component discovery and creation.
Supports multiple named instances of each component type
(data, detector, validator, entry, exit) with cross-referencing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import reduce
from threading import Event
from typing import TYPE_CHECKING, Any, Self

import polars as pl

from signalflow.api.exceptions import (
    DetectorNotFoundError,
    DuplicateComponentNameError,
    InvalidParameterError,
    MissingDataError,
    MissingDetectorError,
    ValidatorNotFoundError,
)
from signalflow.core import (
    RawData,
    SfComponentType,
    Signals,
    default_registry,
)

if TYPE_CHECKING:
    from signalflow.api.result import BacktestResult
    from signalflow.detector.base import SignalDetector
    from signalflow.validator.base import SignalValidator


@dataclass
class BacktestBuilder:
    """
    Fluent builder for backtest configuration.

    Supports both single-component (backward compatible) and multi-component
    configuration with named instances and cross-referencing.

    Example (simple):
        >>> import signalflow as sf
        >>>
        >>> result = (
        ...     sf.Backtest("my_strategy")
        ...     .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        ...     .detector("example/sma_cross", fast_period=20, slow_period=50)
        ...     .entry(size_pct=0.1, max_positions=5)
        ...     .exit(tp=0.03, sl=0.015, trailing=0.02)
        ...     .capital(50_000)
        ...     .run()
        ... )

    Example (multi-component):
        >>> result = (
        ...     sf.Backtest("ensemble")
        ...     .data(raw=spot_1m, name="1m")
        ...     .data(raw=spot_1h, name="1h")
        ...     .detector("sma_cross", name="trend", data_source="1h")
        ...     .detector("volume_spike", name="volume", data_source="1m")
        ...     .aggregation(mode="weighted", weights=[0.7, 0.3])
        ...     .entry(name="trend_entry", source_detector="trend", size_pct=0.15)
        ...     .entry(name="volume_entry", source_detector="volume", size_pct=0.05)
        ...     .exit(name="standard", tp=0.03, sl=0.015)
        ...     .exit(name="trailing", trailing=0.02)
        ...     .capital(50_000)
        ...     .run()
        ... )
    """

    strategy_id: str = "backtest"

    # Internal state — single mode (backward compat)
    _raw: RawData | None = field(default=None, repr=False)
    _detector: SignalDetector | None = field(default=None, repr=False)
    _signals: Signals | None = field(default=None, repr=False)
    _entry_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _exit_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _capital: float = 10_000.0
    _fee: float = 0.001
    _show_progress: bool = True
    _data_params: dict[str, Any] | None = field(default=None, repr=False)

    # Multi-component state (new)
    _named_data: dict[str, RawData | dict[str, Any]] = field(default_factory=dict, repr=False)
    _named_detectors: dict[str, SignalDetector] = field(default_factory=dict, repr=False)
    _named_validators: dict[str, SignalValidator] = field(default_factory=dict, repr=False)
    _named_entries: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _named_exits: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _detector_data_sources: dict[str, str] = field(default_factory=dict, repr=False)
    _aggregation_config: dict[str, Any] | None = field(default=None, repr=False)

    # =========================================================================
    # Data Configuration
    # =========================================================================

    def data(
        self,
        raw: RawData | None = None,
        *,
        name: str | None = None,
        exchange: str | None = None,
        pairs: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        timeframe: str = "1m",
        data_type: str = "perpetual",
    ) -> Self:
        """
        Configure data source. Can be called multiple times with different names.

        Args:
            raw: Pre-loaded RawData instance
            name: Unique name for this data source (for cross-referencing)
            exchange: Exchange name ("binance", "okx", "bybit")
            pairs: List of trading pairs
            start: Start date (ISO string or datetime)
            end: End date (default: now)
            timeframe: Candle timeframe (default: "1m")
            data_type: Data type ("spot", "futures", "perpetual")

        Returns:
            Self for method chaining

        Examples:
            >>> .data(raw=my_raw_data)
            >>> .data(raw=spot_1m, name="1m")
            >>> .data(raw=spot_1h, name="1h")
        """
        if name is not None:
            if name in self._named_data:
                raise DuplicateComponentNameError("data source", name)
            if raw is not None:
                self._named_data[name] = raw
            else:
                self._named_data[name] = {
                    "exchange": exchange,
                    "pairs": pairs,
                    "start": start,
                    "end": end,
                    "timeframe": timeframe,
                    "data_type": data_type,
                }
        else:
            # Backward compatible single-mode
            if raw is not None:
                self._raw = raw
            else:
                self._data_params = {
                    "exchange": exchange,
                    "pairs": pairs,
                    "start": start,
                    "end": end,
                    "timeframe": timeframe,
                    "data_type": data_type,
                }
        return self

    def datas(
        self,
        items: list[tuple[str, RawData]] | dict[str, RawData],
    ) -> Self:
        """
        Set multiple data sources at once.

        Args:
            items: List of (name, RawData) tuples or dict of name -> RawData

        Returns:
            Self for method chaining
        """
        if isinstance(items, dict):
            items = list(items.items())
        for name, raw in items:
            self.data(raw=raw, name=name)
        return self

    # =========================================================================
    # Detector Configuration
    # =========================================================================

    def detector(
        self,
        detector: SignalDetector | str,
        *,
        name: str | None = None,
        data_source: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Add a signal detector. Can be called multiple times to add multiple detectors.

        Args:
            detector: SignalDetector instance OR registry name (e.g., "example/sma_cross")
            name: Unique name for cross-referencing (auto-generated if omitted)
            data_source: Name of data source to use (from .data(name=...))
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining

        Examples:
            >>> .detector("example/sma_cross", fast_period=20)
            >>> .detector("sma_cross", name="trend", data_source="1h")
            >>> .detector("volume_spike", name="volume", data_source="1m")
        """
        # Resolve detector instance
        instance = self._resolve_detector_instance(detector, **kwargs)

        # Determine name
        if name is None:
            name = "default" if not self._named_detectors else f"detector_{len(self._named_detectors)}"

        if name in self._named_detectors:
            raise DuplicateComponentNameError("detector", name)

        self._named_detectors[name] = instance

        # Track data source mapping
        if data_source is not None:
            self._detector_data_sources[name] = data_source

        # Backward compat: keep _detector pointing to last one
        self._detector = instance

        return self

    def detectors(
        self,
        items: list[tuple[str, SignalDetector | str] | SignalDetector | str],
        **kwargs: Any,
    ) -> Self:
        """
        Set multiple detectors at once.

        Args:
            items: List of (name, detector) tuples or just detectors.
            **kwargs: Shared parameters for registry-based creation

        Returns:
            Self for method chaining

        Examples:
            >>> .detectors([("trend", "sma_cross"), ("volume", "volume_spike")])
            >>> .detectors([detector1, detector2])
        """
        for item in items:
            if isinstance(item, tuple):
                item_name, det = item
                self.detector(det, name=item_name, **kwargs)
            else:
                self.detector(item, **kwargs)
        return self

    def signals(self, signals: Signals) -> Self:
        """
        Use pre-computed signals (skip detection).

        Args:
            signals: Pre-computed Signals instance

        Returns:
            Self for method chaining
        """
        self._signals = signals
        return self

    def aggregation(
        self,
        mode: str = "merge",
        *,
        min_agreement: float = 0.5,
        weights: list[float] | None = None,
        probability_threshold: float = 0.5,
    ) -> Self:
        """
        Configure how signals from multiple detectors are combined.

        Args:
            mode: Aggregation mode ("merge", "majority", "weighted",
                  "unanimous", "any", "meta_labeling")
            min_agreement: Minimum detector agreement fraction (for majority)
            weights: Per-detector weights (for weighted mode)
            probability_threshold: Minimum combined probability

        Returns:
            Self for method chaining
        """
        self._aggregation_config = {
            "mode": mode,
            "min_agreement": min_agreement,
            "weights": weights,
            "probability_threshold": probability_threshold,
        }
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
        Add a signal validator (meta-labeler).

        Args:
            validator: SignalValidator instance OR registry name
            name: Unique name for cross-referencing
            **kwargs: Parameters for registry-based creation

        Returns:
            Self for method chaining

        Examples:
            >>> .validator(LightGBMValidator(n_estimators=200))
            >>> .validator("validator/lightgbm", name="ml_filter")
        """
        if isinstance(validator, str):
            try:
                instance = default_registry.create(
                    SfComponentType.VALIDATOR,
                    validator,
                    **kwargs,
                )
            except KeyError:
                available = default_registry.list(SfComponentType.VALIDATOR)
                raise ValidatorNotFoundError(validator, available) from None
        else:
            instance = validator

        if name is None:
            name = "default" if not self._named_validators else f"validator_{len(self._named_validators)}"

        if name in self._named_validators:
            raise DuplicateComponentNameError("validator", name)

        self._named_validators[name] = instance
        return self

    def validators(
        self,
        items: list[tuple[str, SignalValidator | str] | SignalValidator | str],
        **kwargs: Any,
    ) -> Self:
        """
        Set multiple validators at once.

        Args:
            items: List of (name, validator) tuples or just validators

        Returns:
            Self for method chaining
        """
        for item in items:
            if isinstance(item, tuple):
                item_name, val = item
                self.validator(val, name=item_name, **kwargs)
            else:
                self.validator(item, **kwargs)
        return self

    # =========================================================================
    # Entry Configuration
    # =========================================================================

    def entry(
        self,
        *,
        name: str | None = None,
        rule: str | None = None,
        size: float | None = None,
        size_pct: float | None = None,
        max_positions: int = 10,
        max_per_pair: int = 1,
        source_detector: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Configure entry rules. Can be called multiple times with different names.

        Args:
            name: Unique name for this entry rule
            rule: Registry name for custom entry rule (e.g., "signal")
            size: Fixed position size in quote currency
            size_pct: Position size as % of capital (overrides size)
            max_positions: Maximum total concurrent positions
            max_per_pair: Maximum positions per trading pair
            source_detector: Name of detector whose signals this entry uses
            **kwargs: Additional params for custom rule

        Returns:
            Self for method chaining

        Examples:
            >>> .entry(size_pct=0.1, max_positions=5)
            >>> .entry(name="trend", source_detector="trend", size_pct=0.15)
        """
        config = {
            "rule": rule,
            "size": size,
            "size_pct": size_pct,
            "max_positions": max_positions,
            "max_per_pair": max_per_pair,
            "source_detector": source_detector,
            **kwargs,
        }

        if name is not None:
            if name in self._named_entries:
                raise DuplicateComponentNameError("entry rule", name)
            self._named_entries[name] = config
        else:
            # Backward compat single-mode
            self._entry_config = config
        return self

    def entries(
        self,
        items: list[tuple[str, dict[str, Any]]] | dict[str, dict[str, Any]],
    ) -> Self:
        """
        Set multiple entry rules at once.

        Args:
            items: List of (name, config_dict) tuples or dict of name -> config

        Returns:
            Self for method chaining
        """
        if isinstance(items, dict):
            items = list(items.items())
        for entry_name, config in items:
            self.entry(name=entry_name, **config)
        return self

    # =========================================================================
    # Exit Configuration
    # =========================================================================

    def exit(
        self,
        *,
        name: str | None = None,
        rule: str | None = None,
        tp: float | None = None,
        sl: float | None = None,
        trailing: float | None = None,
        time_limit: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Configure exit rules. Can be called multiple times with different names.

        Args:
            name: Unique name for this exit rule
            rule: Registry name for custom exit rule
            tp: Take profit percentage (e.g., 0.03 = 3%)
            sl: Stop loss percentage (e.g., 0.015 = 1.5%)
            trailing: Trailing stop percentage
            time_limit: Maximum bars to hold position
            **kwargs: Additional params for custom rule

        Returns:
            Self for method chaining

        Examples:
            >>> .exit(tp=0.03, sl=0.015)
            >>> .exit(name="standard", tp=0.03, sl=0.015)
            >>> .exit(name="trailing", trailing=0.02)
        """
        config = {
            "rule": rule,
            "tp": tp,
            "sl": sl,
            "trailing": trailing,
            "time_limit": time_limit,
            **kwargs,
        }

        if name is not None:
            if name in self._named_exits:
                raise DuplicateComponentNameError("exit rule", name)
            self._named_exits[name] = config
        else:
            self._exit_config = config
        return self

    def exits(
        self,
        items: list[tuple[str, dict[str, Any]]] | dict[str, dict[str, Any]],
    ) -> Self:
        """
        Set multiple exit rules at once.

        Args:
            items: List of (name, config_dict) tuples or dict of name -> config

        Returns:
            Self for method chaining
        """
        if isinstance(items, dict):
            items = list(items.items())
        for exit_name, config in items:
            self.exit(name=exit_name, **config)
        return self

    # =========================================================================
    # Other Configuration
    # =========================================================================

    def capital(self, amount: float) -> Self:
        """
        Set initial capital.

        Args:
            amount: Initial capital in quote currency

        Returns:
            Self for method chaining

        Raises:
            InvalidParameterError: If amount is not positive
        """
        if amount <= 0:
            raise InvalidParameterError(
                "capital",
                amount,
                "Capital must be a positive number.",
                hint="Use a value like 10_000 or 50_000.0",
            )
        self._capital = amount
        return self

    def fee(self, rate: float) -> Self:
        """
        Set trading fee rate.

        Args:
            rate: Fee rate as decimal (e.g., 0.001 = 0.1%)

        Returns:
            Self for method chaining
        """
        self._fee = rate
        return self

    def progress(self, show: bool = True) -> Self:
        """
        Enable/disable progress bar during backtest.

        Args:
            show: Whether to show progress bar

        Returns:
            Self for method chaining
        """
        self._show_progress = show
        return self

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Export builder configuration as JSON-serializable dict.

        Captures all configured state so it can be restored via ``from_dict()``.
        RawData instances are NOT serialized — only parameter-based data configs.

        Returns:
            Configuration dict suitable for JSON serialization.
        """
        config: dict[str, Any] = {"strategy_id": self.strategy_id}

        # Data
        if self._data_params:
            config["data"] = self._serialize_data_params(self._data_params)
        if self._named_data:
            config["data_sources"] = {
                name: self._serialize_data_params(src) if isinstance(src, dict) else {"_type": "preloaded"}
                for name, src in self._named_data.items()
            }

        # Detectors
        if self._named_detectors:
            config["detectors"] = {}
            for name, det in self._named_detectors.items():
                det_info: dict[str, Any] = {"class_name": det.__class__.__name__}
                if dataclasses.is_dataclass(det):
                    det_info["params"] = {
                        f.name: getattr(det, f.name)
                        for f in dataclasses.fields(det)
                        if not f.name.startswith("_")
                        and f.name
                        not in (
                            "component_type",
                            "features",
                            "pair_col",
                            "ts_col",
                            "raw_data_type",
                            "require_probability",
                            "keep_only_latest_per_pair",
                            "group_col",
                            "allowed_signal_types",
                            "signal_category",
                        )
                        and self._is_json_serializable(getattr(det, f.name))
                    }
                if name in self._detector_data_sources:
                    det_info["data_source"] = self._detector_data_sources[name]
                config["detectors"][name] = det_info

        # Aggregation
        if self._aggregation_config:
            config["aggregation"] = self._aggregation_config

        # Entry
        if self._entry_config:
            config["entry"] = self._entry_config
        if self._named_entries:
            config["entries"] = dict(self._named_entries)

        # Exit
        if self._exit_config:
            config["exit"] = self._exit_config
        if self._named_exits:
            config["exits"] = dict(self._named_exits)

        # Capital & fees
        config["capital"] = self._capital
        config["fee"] = self._fee

        return config

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BacktestBuilder:
        """Reconstruct a builder from a config dict.

        Supports two config formats:
        1. Builder format (from ``to_dict()``): detectors, entry, exit
        2. Flow config format: detector, strategy.entry_rules, strategy.exit_rules

        Detector instances are recreated from registry using class_name/type + params.
        Data sources marked as ``_type: preloaded`` are skipped — caller must
        re-attach them via ``.data(raw=...)``.

        Args:
            config: Config dict (from ``to_dict()`` or ``load_flow_config()``).

        Returns:
            Configured BacktestBuilder.

        Example:
            >>> # From flow config
            >>> config = sf.config.load("grid_sma")
            >>> builder = sf.Backtest.from_dict(config)

            >>> # From builder config
            >>> builder = sf.Backtest.from_dict(old_builder.to_dict())
        """
        # Determine strategy_id
        strategy_id = config.get("strategy_id", "backtest")
        if "strategy" in config and "strategy_id" in config["strategy"]:
            strategy_id = config["strategy"]["strategy_id"]
        if "flow_id" in config and strategy_id == "backtest":
            strategy_id = config["flow_id"]

        builder = cls(strategy_id=strategy_id)

        # Data (parameter-based only)
        if "data" in config:
            data_cfg = config["data"]
            # Flow config format has nested structure
            if "pairs" in data_cfg and not any(k in data_cfg for k in ["exchange", "source"]):
                # Don't call .data() for flow config - it just has pairs/timeframe
                pass
            else:
                builder.data(**data_cfg)
        if "data_sources" in config:
            for name, src in config["data_sources"].items():
                if isinstance(src, dict) and src.get("_type") == "preloaded":
                    continue
                builder.data(name=name, **src)

        # Detectors - handle both formats
        if "detectors" in config:
            # Builder format: detectors dict
            for name, det_info in config["detectors"].items():
                params = det_info.get("params", {})
                class_name = det_info.get("class_name", name)
                data_source = det_info.get("data_source")
                builder.detector(
                    class_name,
                    name=name,
                    data_source=data_source,
                    **params,
                )
        elif "detector" in config:
            # Flow config format: single detector
            det_cfg = config["detector"]
            det_type = det_cfg.get("type", "")
            det_params = {k: v for k, v in det_cfg.items() if k != "type"}
            if det_type:
                builder.detector(det_type, **det_params)

        # Aggregation
        if "aggregation" in config:
            builder.aggregation(**config["aggregation"])

        # Entry - handle both formats
        if "entry" in config:
            builder.entry(**config["entry"])
        if "entries" in config:
            for name, entry_cfg in config["entries"].items():
                builder.entry(name=name, **entry_cfg)

        # Flow config format: strategy.entry_rules
        if "strategy" in config and "entry_rules" in config["strategy"]:
            entry_rules = config["strategy"]["entry_rules"]
            if entry_rules:
                rule = entry_rules[0]  # Use first rule
                entry_cfg = cls._convert_entry_rule_to_config(rule)
                builder.entry(**entry_cfg)

        # Exit - handle both formats
        if "exit" in config:
            builder.exit(**config["exit"])
        if "exits" in config:
            for name, exit_cfg in config["exits"].items():
                builder.exit(name=name, **exit_cfg)

        # Flow config format: strategy.exit_rules
        if "strategy" in config and "exit_rules" in config["strategy"]:
            exit_rules = config["strategy"]["exit_rules"]
            if exit_rules:
                rule = exit_rules[0]  # Use first rule
                exit_cfg = cls._convert_exit_rule_to_config(rule)
                builder.exit(**exit_cfg)

        # Capital & fees
        if "capital" in config:
            builder.capital(config["capital"])
        if "fee" in config:
            builder.fee(config["fee"])

        return builder

    @classmethod
    def _convert_entry_rule_to_config(cls, rule: dict[str, Any]) -> dict[str, Any]:
        """Convert flow config entry_rule to builder entry config.

        Handles entry_filters by instantiating them from registry.
        """
        config: dict[str, Any] = {
            "size": rule.get("base_position_size", 100.0),
            "max_positions": rule.get("max_total_positions", 10),
            "max_per_pair": rule.get("max_positions_per_pair", 1),
        }

        # Copy additional params
        for key in ["min_probability", "use_probability_sizing"]:
            if key in rule:
                config[key] = rule[key]

        # Handle entry_filters
        if "entry_filters" in rule:
            filters = cls._instantiate_entry_filters(rule["entry_filters"])
            if filters:
                config["entry_filters"] = filters

        return config

    @classmethod
    def _instantiate_entry_filters(cls, filters_config: list[dict[str, Any]]) -> list[Any]:
        """Instantiate entry filters from config.

        Args:
            filters_config: List of filter configs with 'type' and params.

        Returns:
            List of instantiated EntryFilter objects.
        """
        from signalflow.core import SfComponentType, default_registry

        filters = []
        for filter_cfg in filters_config:
            filter_type = filter_cfg.get("type", "")
            if not filter_type:
                continue

            try:
                filter_cls = default_registry.get(SfComponentType.STRATEGY_ENTRY_RULE, filter_type)
                # Extract params (everything except 'type')
                params = {k: v for k, v in filter_cfg.items() if k != "type"}
                # Get valid dataclass fields
                if hasattr(filter_cls, "__dataclass_fields__"):
                    valid_fields = {f for f in filter_cls.__dataclass_fields__ if f != "component_type"}
                    params = {k: v for k, v in params.items() if k in valid_fields}
                filters.append(filter_cls(**params))
            except (KeyError, TypeError) as e:
                from loguru import logger

                logger.warning(f"Could not instantiate entry filter '{filter_type}': {e}")

        return filters

    @classmethod
    def _convert_exit_rule_to_config(cls, rule: dict[str, Any]) -> dict[str, Any]:
        """Convert flow config exit_rule to builder exit config."""
        config: dict[str, Any] = {}
        rule_type = rule.get("type", "tp_sl")

        if rule_type == "tp_sl":
            if "take_profit_pct" in rule:
                config["tp"] = rule["take_profit_pct"]
            if "stop_loss_pct" in rule:
                config["sl"] = rule["stop_loss_pct"]
            if "trailing_stop_pct" in rule:
                config["trailing"] = rule["trailing_stop_pct"]
        else:
            # Copy all params except type
            config = {k: v for k, v in rule.items() if k != "type"}

        return config

    @staticmethod
    def _serialize_data_params(params: dict[str, Any]) -> dict[str, Any]:
        """Serialize data parameters, converting datetimes to ISO strings."""
        result = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            else:
                result[k] = v
        return result

    @staticmethod
    def _is_json_serializable(value: Any) -> bool:
        """Check if a value is JSON-serializable."""
        return isinstance(value, (str, int, float, bool, type(None), list, dict))

    # =========================================================================
    # Execution
    # =========================================================================

    def run(
        self,
        *,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
        cancel_event: Event | None = None,
    ) -> BacktestResult:
        """
        Execute backtest and return results.

        Resolves all configuration, creates components from registry,
        runs the backtest, and wraps results in BacktestResult.

        Args:
            progress_callback: Called every N bars with ``(current, total, metrics)``.
                Useful for streaming progress to UI via WebSocket.
            cancel_event: Threading event — set it to gracefully stop the backtest.
                The runner will break after the current bar and return partial results.

        Returns:
            BacktestResult with trades, metrics, and analytics

        Raises:
            MissingDataError: If data not configured
            MissingDetectorError: If detector/signals not configured
        """
        from signalflow.api.result import BacktestResult

        # 1. Resolve data
        raw = self._resolve_data()

        # 2. Resolve signals (with multi-detector support)
        merged_signals, named_signals = self._resolve_signals(raw)

        # 3. Build components from registry
        entry_rules = self._build_entry_rules()
        exit_rules = self._build_exit_rules()
        broker = self._build_broker()

        # 4. Create and run runner
        runner = self._build_runner(
            broker,
            entry_rules,
            exit_rules,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        state = runner.run(
            raw_data=raw,
            signals=merged_signals,
            named_signals=named_signals if named_signals else None,
        )

        # 5. Wrap in BacktestResult
        return BacktestResult(
            state=state,
            trades=getattr(runner, "trades", []),
            signals=merged_signals,
            raw=raw,
            config={
                "capital": self._capital,
                "fee": self._fee,
                **(self._entry_config if self._entry_config else {}),
                **(self._exit_config if self._exit_config else {}),
            },
            metrics_df=getattr(runner, "metrics_df", None),
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of error/warning messages
        """
        issues: list[str] = []

        # Check data
        has_data = self._raw is not None or self._data_params is not None or bool(self._named_data)
        if not has_data:
            issues.append("ERROR: No data source configured. Use .data()")
        elif self._data_params and not self._named_data:
            if not self._data_params.get("pairs"):
                issues.append("ERROR: No pairs specified in .data()")
            if not self._data_params.get("start"):
                issues.append("ERROR: No start date specified in .data()")

        # Check detector/signals
        has_detector = self._detector is not None or bool(self._named_detectors) or self._signals is not None
        if not has_detector:
            issues.append("ERROR: No detector or signals configured. Use .detector() or .signals()")

        # Check registry availability
        try:
            default_registry.get(SfComponentType.STRATEGY_RUNNER, "backtest")
        except KeyError:
            issues.append("ERROR: BacktestRunner not found in registry")

        # Validate cross-references: entry source_detector must exist
        all_detector_names = set(self._named_detectors.keys())
        for entry_name, config in self._named_entries.items():
            src = config.get("source_detector")
            if src and src not in all_detector_names:
                issues.append(
                    f"ERROR: Entry '{entry_name}' references detector '{src}' "
                    f"which doesn't exist. Available: {sorted(all_detector_names)}"
                )

        # Validate cross-references: detector data_source must exist
        all_data_names = set(self._named_data.keys())
        if all_data_names:
            for det_name, data_src in self._detector_data_sources.items():
                if data_src not in all_data_names:
                    issues.append(
                        f"ERROR: Detector '{det_name}' references data source '{data_src}' "
                        f"which doesn't exist. Available: {sorted(all_data_names)}"
                    )

        # Validate aggregation weights
        if self._aggregation_config:
            weights = self._aggregation_config.get("weights")
            if weights and len(self._named_detectors) > 1 and len(weights) != len(self._named_detectors):
                issues.append(
                    f"ERROR: Aggregation weights length ({len(weights)}) "
                    f"must match detector count ({len(self._named_detectors)})"
                )

        # Validate TP/SL ratio (single mode)
        exit_configs = list(self._named_exits.values()) if self._named_exits else [self._exit_config]
        for cfg in exit_configs:
            tp = cfg.get("tp")
            sl = cfg.get("sl")
            if tp and sl and tp < sl:
                issues.append(f"WARNING: TP ({tp:.1%}) < SL ({sl:.1%}), risk/reward < 1")

        # Validate capital
        if self._capital <= 0:
            issues.append("ERROR: Capital must be positive")

        return issues

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _resolve_detector_instance(self, detector: SignalDetector | str, **kwargs: Any) -> SignalDetector:
        """Resolve a detector from instance or registry name."""
        if isinstance(detector, str):
            try:
                return default_registry.create(
                    SfComponentType.DETECTOR,
                    detector,
                    **kwargs,
                )
            except KeyError:
                available = default_registry.list(SfComponentType.DETECTOR)
                raise DetectorNotFoundError(detector, available) from None
        return detector

    def _resolve_data(self) -> RawData:
        """Load data from params or return pre-loaded."""
        # Multi-mode: combine named data sources into single RawData
        if self._named_data:
            return self._resolve_named_data()

        # Single mode (backward compat)
        if self._raw is not None:
            return self._raw

        if not self._data_params:
            raise MissingDataError()

        from signalflow.api.shortcuts import load

        params = {k: v for k, v in self._data_params.items() if v is not None}
        return load(**params)

    def _resolve_named_data(self) -> RawData:
        """Resolve multiple named data sources into a single RawData."""
        from signalflow.api.shortcuts import load

        resolved: dict[str, pl.DataFrame] = {}
        all_pairs: set[str] = set()
        dt_start = None
        dt_end = None

        for name, source in self._named_data.items():
            if isinstance(source, RawData):
                # Extract the main DataFrame from the RawData
                for key in source.data:
                    df = source.get(key)
                    if df.height > 0:
                        resolved[name] = df
                        break
                all_pairs.update(source.pairs)
                if dt_start is None or source.datetime_start < dt_start:
                    dt_start = source.datetime_start
                if dt_end is None or source.datetime_end > dt_end:
                    dt_end = source.datetime_end
            elif isinstance(source, dict):
                # Lazy load params
                params = {k: v for k, v in source.items() if v is not None}
                raw = load(**params)
                for key in raw.data:
                    df = raw.get(key)
                    if df.height > 0:
                        resolved[name] = df
                        break
                all_pairs.update(raw.pairs)
                if dt_start is None or raw.datetime_start < dt_start:
                    dt_start = raw.datetime_start
                if dt_end is None or raw.datetime_end > dt_end:
                    dt_end = raw.datetime_end

        if not resolved:
            raise MissingDataError()

        # Also keep backward compat — first source is the "spot" key
        first_name = next(iter(resolved))
        data_dict: dict[str, pl.DataFrame | dict[str, pl.DataFrame]] = {
            "spot": resolved[first_name],
        }
        # Add all named sources
        for k, v in resolved.items():
            data_dict[k] = v

        return RawData(
            datetime_start=dt_start or datetime.min,
            datetime_end=dt_end or datetime.max,
            pairs=sorted(all_pairs),
            data=data_dict,
        )

    def _resolve_signals(self, raw: RawData) -> tuple[Signals, dict[str, Signals]]:
        """Detect signals from all detectors and merge."""
        # Pre-computed signals
        if self._signals is not None:
            return self._signals, {}

        # Multi-detector mode
        if len(self._named_detectors) > 1:
            return self._resolve_multi_signals(raw)

        # Single detector mode (backward compat)
        if self._named_detectors:
            # Single named detector — run it
            det_name, detector = next(iter(self._named_detectors.items()))
            signals = self._run_detector(detector, raw, det_name)
            return signals, {det_name: signals}

        if self._detector is not None:
            from signalflow.core.containers.raw_data_view import RawDataView

            signals = self._detector.run(RawDataView(raw=raw))
            return signals, {}

        raise MissingDetectorError()

    def _resolve_multi_signals(self, raw: RawData) -> tuple[Signals, dict[str, Signals]]:
        """Run multiple detectors and aggregate signals."""
        named_signals: dict[str, Signals] = {}

        for det_name, detector in self._named_detectors.items():
            signals = self._run_detector(detector, raw, det_name)
            # Tag signals with source detector name
            signals = Signals(signals.value.with_columns(pl.lit(det_name).alias("_source_detector")))
            named_signals[det_name] = signals

        # Apply validators (if any)
        if self._named_validators:
            named_signals = self._apply_validators(named_signals, raw)

        # Merge signals
        merged = self._merge_signals(named_signals)

        return merged, named_signals

    def _run_detector(self, detector: SignalDetector, raw: RawData, det_name: str) -> Signals:
        """Run a single detector with appropriate data source."""
        from signalflow.core.containers.raw_data_view import RawDataView

        data_source = self._detector_data_sources.get(det_name)
        if data_source and data_source in raw.data:
            # Create a focused RawData with the specific data source as "spot"
            df = raw.get(data_source)
            focused_raw = RawData(
                datetime_start=raw.datetime_start,
                datetime_end=raw.datetime_end,
                pairs=raw.pairs,
                data={"spot": df},
            )
            return detector.run(RawDataView(raw=focused_raw))

        return detector.run(RawDataView(raw=raw))

    def _apply_validators(
        self,
        named_signals: dict[str, Signals],
        raw: RawData,
    ) -> dict[str, Signals]:
        """Apply validators to signals."""
        for _v_name, validator in self._named_validators.items():
            for d_name in named_signals:
                try:
                    # Get features from raw data for validation
                    first_key = next(iter(raw.data.keys()), "spot")
                    features = raw.get(first_key)
                    named_signals[d_name] = validator.validate_signals(named_signals[d_name], features)
                except (NotImplementedError, Exception):
                    # Validator may not be trained yet — skip silently
                    pass
        return named_signals

    def _merge_signals(self, named_signals: dict[str, Signals]) -> Signals:
        """Merge signals from multiple detectors."""
        signals_list = list(named_signals.values())

        if not signals_list:
            return Signals(pl.DataFrame())

        if len(signals_list) == 1:
            return signals_list[0]

        # Use aggregator if configured
        if self._aggregation_config and self._aggregation_config.get("mode") != "merge":
            from signalflow.strategy.component.entry.aggregation import (
                SignalAggregator,
                VotingMode,
            )

            mode = self._aggregation_config["mode"]
            aggregator = SignalAggregator(
                voting_mode=VotingMode(mode),
                min_agreement=self._aggregation_config.get("min_agreement", 0.5),
                weights=self._aggregation_config.get("weights"),
                probability_threshold=self._aggregation_config.get("probability_threshold", 0.5),
            )
            return aggregator.aggregate(
                signals_list,
                detector_names=list(named_signals.keys()),
            )

        # Default: sequential merge via + (list order = priority)
        return reduce(lambda a, b: a + b, signals_list)

    def _build_entry_rules(self) -> list[Any]:
        """Build entry rules from config using registry."""
        if self._named_entries:
            rules = []
            for _name, config in self._named_entries.items():
                rules.append(self._build_single_entry(config))
            return rules

        # Single mode (backward compat)
        return [self._build_single_entry(self._entry_config)]

    def _build_single_entry(self, config: dict[str, Any]) -> Any:
        """Build a single entry rule from config dict."""
        rule_name = config.get("rule") or "signal"

        try:
            rule_cls = default_registry.get(SfComponentType.STRATEGY_ENTRY_RULE, rule_name)
        except (KeyError, AttributeError):
            from signalflow.strategy.component.entry.signal import SignalEntryRule

            rule_cls = SignalEntryRule

        # Calculate position size
        size = config.get("size")
        if size is None:
            size = 100.0
        size_pct = config.get("size_pct")
        if size_pct:
            size = self._capital * size_pct

        rule_kwargs: dict[str, Any] = {
            "base_position_size": size,
            "max_positions_per_pair": config.get("max_per_pair", 1),
            "max_total_positions": config.get("max_positions", 10),
        }
        # Pass entry filters if provided (from graph converter)
        if config.get("entry_filters"):
            rule_kwargs["entry_filters"] = config["entry_filters"]

        rule = rule_cls(**rule_kwargs)

        # Set source_detector for cross-referencing
        source_detector = config.get("source_detector")
        if source_detector and hasattr(rule, "source_detector"):
            rule.source_detector = source_detector

        return rule

    def _build_exit_rules(self) -> list[Any]:
        """Build exit rules from config using registry."""
        if self._named_exits:
            rules: list[Any] = []
            for _name, config in self._named_exits.items():
                rules.extend(self._build_single_exit(config))
            return rules

        # Single mode (backward compat)
        return self._build_single_exit(self._exit_config)

    def _build_single_exit(self, config: dict[str, Any]) -> list[Any]:
        """Build exit rules from a single config dict."""
        rules: list[Any] = []

        # TP/SL rule
        tp = config.get("tp")
        sl = config.get("sl")
        if tp or sl:
            try:
                tpsl_cls = default_registry.get(SfComponentType.STRATEGY_EXIT_RULE, "tp_sl")
            except KeyError:
                from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit

                tpsl_cls = TakeProfitStopLossExit

            rules.append(
                tpsl_cls(
                    take_profit_pct=tp or 0.02,
                    stop_loss_pct=sl or 0.01,
                )
            )

        # Trailing stop
        trailing = config.get("trailing")
        if trailing:
            try:
                trail_cls = default_registry.get(SfComponentType.STRATEGY_EXIT_RULE, "trailing")
                rules.append(trail_cls(trail_pct=trailing))
            except KeyError:
                pass

        # Time-based exit
        time_limit = config.get("time_limit")
        if time_limit:
            try:
                time_cls = default_registry.get(SfComponentType.STRATEGY_EXIT_RULE, "time_based")
                rules.append(time_cls(max_bars=time_limit))
            except KeyError:
                pass

        # Pre-built exit rule instances (from graph converter)
        extra = config.get("_extra_rules")
        if extra:
            rules.extend(extra)

        # Default if nothing configured
        if not rules:
            from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit

            rules.append(TakeProfitStopLossExit(take_profit_pct=0.02, stop_loss_pct=0.01))

        return rules

    def _build_broker(self) -> Any:
        """Build broker from registry."""
        from signalflow.data.strategy_store.memory import InMemoryStrategyStore

        try:
            broker_cls = default_registry.get(SfComponentType.STRATEGY_BROKER, "backtest")
            executor_cls = default_registry.get(SfComponentType.STRATEGY_EXECUTOR, "virtual/spot")

            executor = executor_cls(fee_rate=self._fee)
            return broker_cls(executor=executor, store=InMemoryStrategyStore())
        except KeyError:
            # Fallback to direct imports
            from signalflow.strategy.broker import BacktestBroker
            from signalflow.strategy.broker.executor import VirtualSpotExecutor

            return BacktestBroker(
                executor=VirtualSpotExecutor(fee_rate=self._fee),
                store=InMemoryStrategyStore(),
            )

    def _build_runner(
        self,
        broker: Any,
        entry_rules: list[Any],
        exit_rules: list[Any],
        *,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
        cancel_event: Event | None = None,
    ) -> Any:
        """Build runner from registry."""
        try:
            runner_cls = default_registry.get(SfComponentType.STRATEGY_RUNNER, "backtest")
        except KeyError:
            from signalflow.strategy.runner import BacktestRunner

            runner_cls = BacktestRunner

        return runner_cls(
            strategy_id=self.strategy_id,
            broker=broker,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            initial_capital=self._capital,
            show_progress=self._show_progress,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def visualize(
        self,
        *,
        output: str | None = None,
        format: str = "html",
        show: bool = True,
    ) -> str:
        """
        Visualize the configured pipeline.

        Opens an interactive HTML visualization showing the data flow
        from data sources through features to detector and runner.

        Args:
            output: Output file path (optional)
            format: Output format ("html" or "mermaid")
            show: Open in browser (HTML only)

        Returns:
            Rendered output string

        Example:
            >>> sf.Backtest("test").data(...).detector(...).visualize()
        """
        from signalflow import viz

        return viz.pipeline(self, output=output, format=format, show=show)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        parts = [f"strategy_id={self.strategy_id!r}"]
        if self._named_detectors:
            parts.append(f"detectors={list(self._named_detectors.keys())}")
        if self._named_entries:
            parts.append(f"entries={list(self._named_entries.keys())}")
        if self._named_exits:
            parts.append(f"exits={list(self._named_exits.keys())}")
        return f"BacktestBuilder({', '.join(parts)})"


def Backtest(strategy_id: str = "backtest") -> BacktestBuilder:
    """
    Create a new backtest builder.

    This is the recommended way to configure and run backtests.

    Args:
        strategy_id: Unique identifier for the strategy

    Returns:
        BacktestBuilder instance for fluent configuration

    Example:
        >>> result = (
        ...     sf.Backtest("my_strategy")
        ...     .data(exchange="binance", pairs=["BTCUSDT"], start="2024-01-01")
        ...     .detector("example/sma_cross")
        ...     .run()
        ... )
    """
    return BacktestBuilder(strategy_id=strategy_id)
