"""
YAML configuration loader for SignalFlow CLI.

Supports loading backtest configuration from YAML files with validation.
Supports both single-component (backward compatible) and multi-component
configuration with named instances and cross-referencing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Data source configuration."""

    source: str
    pairs: list[str]
    start: str | datetime
    end: str | datetime | None = None
    timeframe: str = "1m"
    data_type: str = "perpetual"


@dataclass
class DetectorConfig:
    """Detector configuration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    data_source: str | None = None


@dataclass
class ValidatorConfig:
    """Validator configuration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntryConfig:
    """Entry rules configuration."""

    rule: str | None = None
    size: float | None = None
    size_pct: float | None = None
    max_positions: int = 10
    max_per_pair: int = 1
    source_detector: str | None = None


@dataclass
class ExitConfig:
    """Exit rules configuration."""

    rule: str | None = None
    tp: float | None = None
    sl: float | None = None
    trailing: float | None = None
    time_limit: int | None = None


@dataclass
class AggregationConfig:
    """Signal aggregation configuration."""

    mode: str = "merge"
    min_agreement: float = 0.5
    weights: list[float] | None = None
    probability_threshold: float = 0.5


@dataclass
class BacktestConfig:
    """
    Complete backtest configuration from YAML.

    Supports both singular (backward compatible) and plural keys:
    - ``data`` / ``data_sources``
    - ``detector`` / ``detectors``
    - ``entry`` / ``entries``
    - ``exit`` / ``exits``

    Example YAML (single):
        ```yaml
        strategy:
          id: my_strategy
        data:
          source: data/binance.duckdb
          pairs: [BTCUSDT]
          start: "2024-01-01"
        detector:
          name: example/sma_cross
          params: {fast_period: 20, slow_period: 50}
        entry:
          size_pct: 0.1
        exit:
          tp: 0.03
          sl: 0.015
        capital: 50000
        ```

    Example YAML (multi-component):
        ```yaml
        strategy:
          id: ensemble
        data_sources:
          spot_1m:
            source: data/binance.duckdb
            pairs: [BTCUSDT]
            start: "2024-01-01"
            timeframe: 1m
          spot_1h:
            source: data/binance.duckdb
            pairs: [BTCUSDT]
            start: "2024-01-01"
            timeframe: 1h
        detectors:
          trend:
            name: example/sma_cross
            params: {fast_period: 20, slow_period: 50}
            data_source: spot_1h
          volume:
            name: example/volume_spike
            params: {threshold: 2.0}
            data_source: spot_1m
        aggregation:
          mode: weighted
          weights: [0.7, 0.3]
        entries:
          trend_entry:
            size_pct: 0.15
            source_detector: trend
          volume_entry:
            size_pct: 0.05
            source_detector: volume
        exits:
          standard:
            tp: 0.03
            sl: 0.015
          trailing:
            trailing: 0.02
        capital: 50000
        ```
    """

    strategy_id: str = "backtest"
    # Singular (backward compat)
    data: DataConfig | None = None
    detector: DetectorConfig | None = None
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    # Plural (new)
    data_sources: dict[str, DataConfig] | None = None
    detectors: dict[str, DetectorConfig] | None = None
    validators: dict[str, ValidatorConfig] | None = None
    entries: dict[str, EntryConfig] | None = None
    exits: dict[str, ExitConfig] | None = None
    aggregation: AggregationConfig | None = None
    # Common
    capital: float = 10_000.0
    fee: float = 0.001
    show_progress: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestConfig:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            BacktestConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BacktestConfig:
        """
        Create config from dictionary.

        Args:
            d: Configuration dictionary

        Returns:
            BacktestConfig instance
        """
        # === Data: singular or plural ===
        data_cfg = None
        data_sources = None
        if "data_sources" in d and d["data_sources"]:
            data_sources = cls._parse_named_data(d["data_sources"])
        elif "data" in d and d["data"]:
            data_cfg = DataConfig(**d["data"])

        # === Detector: singular or plural ===
        detector_cfg = None
        detectors = None
        if "detectors" in d and d["detectors"]:
            detectors = cls._parse_named_detectors(d["detectors"])
        elif "detector" in d and d["detector"]:
            det_data = d["detector"]
            detector_cfg = DetectorConfig(
                name=det_data.get("name", ""),
                params=det_data.get("params", {}),
                data_source=det_data.get("data_source"),
            )

        # === Validators (new, plural only) ===
        validators_cfg = None
        if "validators" in d and d["validators"]:
            validators_cfg = cls._parse_named_validators(d["validators"])

        # === Entry: singular or plural ===
        entry_cfg = EntryConfig()
        entries = None
        if "entries" in d and d["entries"]:
            entries = cls._parse_named_entries(d["entries"])
        elif "entry" in d and d["entry"]:
            entry_data = d["entry"]
            entry_cfg = EntryConfig(
                rule=entry_data.get("rule"),
                size=entry_data.get("size"),
                size_pct=entry_data.get("size_pct"),
                max_positions=entry_data.get("max_positions", 10),
                max_per_pair=entry_data.get("max_per_pair", 1),
                source_detector=entry_data.get("source_detector"),
            )

        # === Exit: singular or plural ===
        exit_cfg = ExitConfig()
        exits = None
        if "exits" in d and d["exits"]:
            exits = cls._parse_named_exits(d["exits"])
        elif "exit" in d and d["exit"]:
            exit_data = d["exit"]
            exit_cfg = ExitConfig(
                rule=exit_data.get("rule"),
                tp=exit_data.get("tp"),
                sl=exit_data.get("sl"),
                trailing=exit_data.get("trailing"),
                time_limit=exit_data.get("time_limit"),
            )

        # === Aggregation ===
        agg_cfg = None
        if "aggregation" in d and d["aggregation"]:
            agg_data = d["aggregation"]
            agg_cfg = AggregationConfig(
                mode=agg_data.get("mode", "merge"),
                min_agreement=agg_data.get("min_agreement", 0.5),
                weights=agg_data.get("weights"),
                probability_threshold=agg_data.get("probability_threshold", 0.5),
            )

        strategy_section = d.get("strategy", {})

        return cls(
            strategy_id=strategy_section.get("id", d.get("strategy_id", "backtest")),
            data=data_cfg,
            detector=detector_cfg,
            entry=entry_cfg,
            exit=exit_cfg,
            data_sources=data_sources,
            detectors=detectors,
            validators=validators_cfg,
            entries=entries,
            exits=exits,
            aggregation=agg_cfg,
            capital=d.get("capital", 10_000.0),
            fee=d.get("fee", 0.001),
            show_progress=d.get("show_progress", True),
        )

    # === Named config parsers ===

    @staticmethod
    def _parse_named_data(raw: dict | list) -> dict[str, DataConfig]:
        """Parse named data configs from dict or list format."""
        if isinstance(raw, list):
            return {item["name"]: DataConfig(**{k: v for k, v in item.items() if k != "name"}) for item in raw}
        return {name: DataConfig(**cfg) for name, cfg in raw.items()}

    @staticmethod
    def _parse_named_detectors(raw: dict | list) -> dict[str, DetectorConfig]:
        """Parse named detector configs from dict or list format."""
        if isinstance(raw, list):
            return {
                item["name"]: DetectorConfig(
                    name=item.get("detector", item.get("name", "")),
                    params=item.get("params", {}),
                    data_source=item.get("data_source"),
                )
                for item in raw
            }
        result = {}
        for key, cfg in raw.items():
            result[key] = DetectorConfig(
                name=cfg.get("name", cfg.get("detector", key)),
                params=cfg.get("params", {}),
                data_source=cfg.get("data_source"),
            )
        return result

    @staticmethod
    def _parse_named_validators(raw: dict | list) -> dict[str, ValidatorConfig]:
        """Parse named validator configs from dict or list format."""
        if isinstance(raw, list):
            return {
                item["name"]: ValidatorConfig(
                    name=item.get("validator", item.get("name", "")),
                    params=item.get("params", {}),
                )
                for item in raw
            }
        result = {}
        for key, cfg in raw.items():
            result[key] = ValidatorConfig(
                name=cfg.get("name", cfg.get("validator", key)),
                params=cfg.get("params", {}),
            )
        return result

    @staticmethod
    def _parse_named_entries(raw: dict | list) -> dict[str, EntryConfig]:
        """Parse named entry configs from dict or list format."""
        if isinstance(raw, list):
            return {
                item["name"]: EntryConfig(
                    rule=item.get("rule"),
                    size=item.get("size"),
                    size_pct=item.get("size_pct"),
                    max_positions=item.get("max_positions", 10),
                    max_per_pair=item.get("max_per_pair", 1),
                    source_detector=item.get("source_detector"),
                )
                for item in raw
            }
        result = {}
        for key, cfg in raw.items():
            result[key] = EntryConfig(
                rule=cfg.get("rule"),
                size=cfg.get("size"),
                size_pct=cfg.get("size_pct"),
                max_positions=cfg.get("max_positions", 10),
                max_per_pair=cfg.get("max_per_pair", 1),
                source_detector=cfg.get("source_detector"),
            )
        return result

    @staticmethod
    def _parse_named_exits(raw: dict | list) -> dict[str, ExitConfig]:
        """Parse named exit configs from dict or list format."""
        if isinstance(raw, list):
            return {
                item["name"]: ExitConfig(
                    rule=item.get("rule"),
                    tp=item.get("tp"),
                    sl=item.get("sl"),
                    trailing=item.get("trailing"),
                    time_limit=item.get("time_limit"),
                )
                for item in raw
            }
        result = {}
        for key, cfg in raw.items():
            result[key] = ExitConfig(
                rule=cfg.get("rule"),
                tp=cfg.get("tp"),
                sl=cfg.get("sl"),
                trailing=cfg.get("trailing"),
                time_limit=cfg.get("time_limit"),
            )
        return result

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of error/warning messages
        """
        issues: list[str] = []

        # Mutual exclusion checks
        if self.data and self.data_sources:
            issues.append("ERROR: Cannot use both 'data' and 'data_sources'. Choose one.")
        if self.detector and self.detectors:
            issues.append("ERROR: Cannot use both 'detector' and 'detectors'. Choose one.")
        if self.entry.rule is not None and self.entries:
            issues.append("ERROR: Cannot use both 'entry' and 'entries'. Choose one.")
        if self.exit.tp is not None and self.exits:
            issues.append("ERROR: Cannot use both 'exit' and 'exits'. Choose one.")

        # Check data
        has_data = self.data is not None or bool(self.data_sources)
        if not has_data:
            issues.append("ERROR: No data source configured")
        elif self.data:
            if not self.data.source:
                issues.append("ERROR: data.source is required")
            if not self.data.pairs:
                issues.append("ERROR: data.pairs is required")
            if not self.data.start:
                issues.append("ERROR: data.start is required")
            from signalflow.data.resample import TIMEFRAME_MINUTES

            if self.data.timeframe not in TIMEFRAME_MINUTES:
                valid = ", ".join(sorted(TIMEFRAME_MINUTES, key=lambda k: TIMEFRAME_MINUTES[k]))
                issues.append(f"ERROR: Unknown timeframe {self.data.timeframe!r}. Valid: {valid}")

        # Check detector
        has_detector = self.detector is not None or bool(self.detectors)
        if not has_detector:
            issues.append("ERROR: No detector configured")
        elif self.detector and not self.detector.name:
            issues.append("ERROR: detector.name is required")

        # Cross-reference validation
        if self.entries and self.detectors:
            det_names = set(self.detectors.keys())
            for entry_name, entry_cfg in self.entries.items():
                if entry_cfg.source_detector and entry_cfg.source_detector not in det_names:
                    issues.append(
                        f"ERROR: Entry '{entry_name}' references detector "
                        f"'{entry_cfg.source_detector}' which doesn't exist. "
                        f"Available: {sorted(det_names)}"
                    )

        if self.detectors and self.data_sources:
            data_names = set(self.data_sources.keys())
            for det_name, det_cfg in self.detectors.items():
                if det_cfg.data_source and det_cfg.data_source not in data_names:
                    issues.append(
                        f"ERROR: Detector '{det_name}' references data source "
                        f"'{det_cfg.data_source}' which doesn't exist. "
                        f"Available: {sorted(data_names)}"
                    )

        # Aggregation weights validation
        if self.aggregation and self.detectors:
            if self.aggregation.weights:
                if len(self.aggregation.weights) != len(self.detectors):
                    issues.append(
                        f"ERROR: Aggregation weights length ({len(self.aggregation.weights)}) "
                        f"must match detector count ({len(self.detectors)})"
                    )

        # Validate TP/SL ratio
        exit_configs: list[ExitConfig] = []
        if self.exits:
            exit_configs = list(self.exits.values())
        else:
            exit_configs = [self.exit]
        for cfg in exit_configs:
            if cfg.tp and cfg.sl and cfg.tp < cfg.sl:
                issues.append(f"WARNING: TP ({cfg.tp:.1%}) < SL ({cfg.sl:.1%}), risk/reward < 1")

        # Validate capital
        if self.capital <= 0:
            issues.append("ERROR: capital must be positive")

        return issues

    def to_builder(self):
        """
        Convert config to BacktestBuilder.

        Returns:
            Configured BacktestBuilder instance
        """
        from signalflow.api.builder import Backtest

        builder = Backtest(self.strategy_id)

        # === Data ===
        if self.data_sources:
            for name, cfg in self.data_sources.items():
                builder.data(
                    name=name,
                    exchange=None,
                    pairs=cfg.pairs,
                    start=cfg.start,
                    end=cfg.end,
                    timeframe=cfg.timeframe,
                    data_type=cfg.data_type,
                )
        elif self.data:
            builder.data(
                exchange=None,
                pairs=self.data.pairs,
                start=self.data.start,
                end=self.data.end,
                timeframe=self.data.timeframe,
                data_type=self.data.data_type,
            )
            if self.data.source:
                from signalflow.api.shortcuts import load

                try:
                    raw = load(
                        source=self.data.source,
                        pairs=self.data.pairs,
                        start=self.data.start,
                        end=self.data.end,
                        timeframe=self.data.timeframe,
                        data_type=self.data.data_type,
                    )
                    builder.data(raw=raw)
                except Exception:
                    pass

        # === Detectors ===
        if self.detectors:
            for name, cfg in self.detectors.items():
                builder.detector(
                    cfg.name,
                    name=name,
                    data_source=cfg.data_source,
                    **cfg.params,
                )
        elif self.detector:
            builder.detector(self.detector.name, **self.detector.params)

        # === Validators ===
        if self.validators:
            for name, cfg in self.validators.items():
                builder.validator(cfg.name, name=name, **cfg.params)

        # === Entry ===
        if self.entries:
            for name, cfg in self.entries.items():
                builder.entry(
                    name=name,
                    rule=cfg.rule,
                    size=cfg.size,
                    size_pct=cfg.size_pct,
                    max_positions=cfg.max_positions,
                    max_per_pair=cfg.max_per_pair,
                    source_detector=cfg.source_detector,
                )
        else:
            builder.entry(
                rule=self.entry.rule,
                size=self.entry.size,
                size_pct=self.entry.size_pct,
                max_positions=self.entry.max_positions,
                max_per_pair=self.entry.max_per_pair,
                source_detector=self.entry.source_detector,
            )

        # === Exit ===
        if self.exits:
            for name, cfg in self.exits.items():
                builder.exit(
                    name=name,
                    rule=cfg.rule,
                    tp=cfg.tp,
                    sl=cfg.sl,
                    trailing=cfg.trailing,
                    time_limit=cfg.time_limit,
                )
        else:
            builder.exit(
                rule=self.exit.rule,
                tp=self.exit.tp,
                sl=self.exit.sl,
                trailing=self.exit.trailing,
                time_limit=self.exit.time_limit,
            )

        # === Aggregation ===
        if self.aggregation:
            builder.aggregation(
                mode=self.aggregation.mode,
                min_agreement=self.aggregation.min_agreement,
                weights=self.aggregation.weights,
                probability_threshold=self.aggregation.probability_threshold,
            )

        # === Other ===
        builder.capital(self.capital)
        builder.fee(self.fee)
        builder.progress(self.show_progress)

        return builder


def generate_sample_config() -> str:
    """
    Generate sample YAML configuration.

    Returns:
        Sample YAML config string
    """
    return """# SignalFlow Backtest Configuration
# ===================================
# Run with: sf run config.yaml

strategy:
  id: my_strategy

# Data source configuration
data:
  source: data/binance.duckdb   # Path to DuckDB file
  pairs:
    - BTCUSDT
    - ETHUSDT
  start: "2024-01-01"
  end: "2024-06-01"             # Optional, defaults to now
  timeframe: 1m                 # Candle timeframe
  data_type: perpetual          # spot | futures | perpetual

# Detector configuration
detector:
  name: example/sma_cross       # Registry name
  params:
    fast_period: 20
    slow_period: 50

# Entry rules
entry:
  size_pct: 0.1                 # 10% of capital per trade
  max_positions: 5              # Max concurrent positions
  max_per_pair: 1               # Max positions per pair

# Exit rules
exit:
  tp: 0.03                      # Take profit: 3%
  sl: 0.015                     # Stop loss: 1.5%
  # trailing: 0.02              # Optional trailing stop
  # time_limit: 100             # Optional max bars to hold

# Capital and fees
capital: 50000
fee: 0.001                      # 0.1% trading fee

# ===================================
# Multi-component configuration
# ===================================
# Use plural keys for multiple components:
#
# data_sources:
#   spot_1m:
#     source: data/binance.duckdb
#     pairs: [BTCUSDT]
#     start: "2024-01-01"
#     timeframe: 1m
#   spot_1h:
#     source: data/binance.duckdb
#     pairs: [BTCUSDT]
#     start: "2024-01-01"
#     timeframe: 1h
#
# detectors:
#   trend:
#     name: example/sma_cross
#     params: {fast_period: 20, slow_period: 50}
#     data_source: spot_1h
#   volume:
#     name: example/volume_spike
#     params: {threshold: 2.0}
#     data_source: spot_1m
#
# validators:
#   ml_filter:
#     name: validator/lightgbm
#     params: {n_estimators: 200}
#
# aggregation:
#   mode: weighted              # merge | majority | weighted | unanimous | any
#   weights: [0.7, 0.3]
#
# entries:
#   trend_entry:
#     size_pct: 0.15
#     source_detector: trend
#   volume_entry:
#     size_pct: 0.05
#     source_detector: volume
#
# exits:
#   standard:
#     tp: 0.03
#     sl: 0.015
#   trailing:
#     trailing: 0.02
"""
