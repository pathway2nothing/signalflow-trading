"""Flow configuration dataclass.

Provides typed access to flow configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectorConfig:
    """Detector configuration.

    Attributes:
        type: Registry name (e.g., 'example/sma_cross')
        params: Additional detector parameters
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectorConfig:
        """Create from dict."""
        type_ = data.get("type", "")
        params = {k: v for k, v in data.items() if k != "type"}
        return cls(type=type_, params=params)


@dataclass
class EntryFilterConfig:
    """Entry filter configuration.

    Attributes:
        type: Registry name (e.g., 'price_distance_filter')
        params: Filter parameters
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntryFilterConfig:
        """Create from dict."""
        type_ = data.get("type", "")
        params = {k: v for k, v in data.items() if k != "type"}
        return cls(type=type_, params=params)


@dataclass
class EntryRuleConfig:
    """Entry rule configuration.

    Attributes:
        type: Rule type (e.g., 'signal')
        base_position_size: Base position size
        max_positions_per_pair: Max positions per pair
        max_total_positions: Max total positions
        entry_filters: List of entry filter configs
        params: Additional parameters
    """

    type: str = "signal"
    base_position_size: float = 100.0
    max_positions_per_pair: int = 1
    max_total_positions: int = 10
    entry_filters: list[EntryFilterConfig] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntryRuleConfig:
        """Create from dict."""
        filters_data = data.get("entry_filters", [])
        filters = [EntryFilterConfig.from_dict(f) for f in filters_data]

        return cls(
            type=data.get("type", "signal"),
            base_position_size=data.get("base_position_size", 100.0),
            max_positions_per_pair=data.get("max_positions_per_pair", 1),
            max_total_positions=data.get("max_total_positions", 10),
            entry_filters=filters,
            params={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "type",
                    "base_position_size",
                    "max_positions_per_pair",
                    "max_total_positions",
                    "entry_filters",
                }
            },
        )


@dataclass
class ExitRuleConfig:
    """Exit rule configuration.

    Attributes:
        type: Rule type (e.g., 'tp_sl')
        params: Exit rule parameters
    """

    type: str = "tp_sl"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExitRuleConfig:
        """Create from dict."""
        type_ = data.get("type", "tp_sl")
        params = {k: v for k, v in data.items() if k != "type"}
        return cls(type=type_, params=params)


@dataclass
class StrategyConfig:
    """Strategy configuration.

    Attributes:
        strategy_id: Strategy identifier
        entry_rules: List of entry rule configs
        exit_rules: List of exit rule configs
        metrics: List of metric types to compute
    """

    strategy_id: str = "backtest"
    entry_rules: list[EntryRuleConfig] = field(default_factory=list)
    exit_rules: list[ExitRuleConfig] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyConfig:
        """Create from dict."""
        entry_rules = [EntryRuleConfig.from_dict(e) for e in data.get("entry_rules", [])]
        exit_rules = [ExitRuleConfig.from_dict(e) for e in data.get("exit_rules", [])]
        metrics = [m.get("type", m) if isinstance(m, dict) else m for m in data.get("metrics", [])]

        return cls(
            strategy_id=data.get("strategy_id", "backtest"),
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            metrics=metrics,
        )


@dataclass
class DataConfig:
    """Data configuration.

    Attributes:
        pairs: List of trading pairs
        timeframe: Timeframe (e.g., '1h', '4h')
        store: Data store config
        period: Time period config
    """

    pairs: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframe: str = "1h"
    store: dict[str, Any] = field(default_factory=dict)
    period: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Create from dict."""
        pairs = data.get("pairs", ["BTCUSDT"])
        if isinstance(pairs, str):
            pairs = [pairs]
        return cls(
            pairs=pairs,
            timeframe=data.get("timeframe", "1h"),
            store=data.get("store", {}),
            period=data.get("period", {}),
        )


@dataclass
class FlowConfig:
    """Complete flow configuration.

    Provides typed access to all flow settings.

    Attributes:
        flow_id: Unique flow identifier
        flow_name: Human-readable name
        description: Flow description
        data: Data configuration
        detector: Detector configuration
        strategy: Strategy configuration
        capital: Initial capital
        fee: Trading fee percentage
        output: Output paths configuration
        telegram: Telegram notification config
        raw: Original raw config dict

    Example:
        >>> config = FlowConfig.from_dict(sf.config.load("grid_sma"))
        >>> config.flow_id
        'grid_sma'
        >>> config.detector.type
        'example/sma_cross'
        >>> config.strategy.entry_rules[0].entry_filters[0].type
        'price_distance_filter'
    """

    flow_id: str
    flow_name: str = ""
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    detector: DetectorConfig | None = None
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    capital: float = 10000.0
    fee: float = 0.001
    output: dict[str, str] = field(default_factory=dict)
    telegram: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowConfig:
        """Create FlowConfig from raw config dict.

        Args:
            data: Raw configuration dictionary (e.g., from load_flow_config)

        Returns:
            Typed FlowConfig instance
        """
        detector = None
        if "detector" in data:
            detector = DetectorConfig.from_dict(data["detector"])

        strategy = StrategyConfig()
        if "strategy" in data:
            strategy = StrategyConfig.from_dict(data["strategy"])

        data_config = DataConfig()
        if "data" in data:
            data_config = DataConfig.from_dict(data["data"])

        return cls(
            flow_id=data.get("flow_id", "unknown"),
            flow_name=data.get("flow_name", data.get("flow_id", "")),
            description=data.get("description", ""),
            data=data_config,
            detector=detector,
            strategy=strategy,
            capital=data.get("capital", 10000.0),
            fee=data.get("fee", 0.001),
            output=data.get("output", {}),
            telegram=data.get("telegram", {}),
            raw=data,
        )

    def to_backtest_config(self) -> dict[str, Any]:
        """Convert to BacktestBuilder.from_dict() compatible format.

        Returns:
            Config dict for BacktestBuilder.from_dict()
        """
        # Use strategy_id if explicitly set, otherwise fall back to flow_id
        strategy_id = self.flow_id
        if self.strategy.strategy_id and self.strategy.strategy_id != "backtest":
            strategy_id = self.strategy.strategy_id

        config: dict[str, Any] = {
            "strategy_id": strategy_id,
            "capital": self.capital,
            "fee": self.fee,
        }

        # Detector
        if self.detector:
            config["detectors"] = {
                "main": {
                    "class_name": self.detector.type,
                    "params": self.detector.params,
                }
            }

        # Entry rules
        if self.strategy.entry_rules:
            rule = self.strategy.entry_rules[0]  # Use first rule
            entry_config: dict[str, Any] = {
                "size": rule.base_position_size,
                "max_positions": rule.max_total_positions,
                "max_per_pair": rule.max_positions_per_pair,
                **rule.params,
            }

            # Add entry_filters config
            if rule.entry_filters:
                entry_config["entry_filters"] = [{"type": f.type, **f.params} for f in rule.entry_filters]

            config["entry"] = entry_config

        # Exit rules
        if self.strategy.exit_rules:
            rule = self.strategy.exit_rules[0]  # Use first rule
            exit_config: dict[str, Any] = {}

            # Map common exit params
            if rule.type == "tp_sl":
                exit_config["tp"] = rule.params.get("take_profit_pct", 0.03)
                exit_config["sl"] = rule.params.get("stop_loss_pct", 0.015)
            else:
                exit_config.update(rule.params)

            config["exit"] = exit_config

        return config
