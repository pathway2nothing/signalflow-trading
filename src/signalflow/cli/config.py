"""
YAML configuration loader for SignalFlow CLI.

Supports loading backtest configuration from YAML files with validation.
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
    timeframe: str = "1h"
    data_type: str = "perpetual"


@dataclass
class DetectorConfig:
    """Detector configuration."""

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


@dataclass
class ExitConfig:
    """Exit rules configuration."""

    rule: str | None = None
    tp: float | None = None
    sl: float | None = None
    trailing: float | None = None
    time_limit: int | None = None


@dataclass
class BacktestConfig:
    """
    Complete backtest configuration from YAML.

    Example YAML:
        ```yaml
        strategy:
          id: my_strategy

        data:
          source: data/binance.duckdb
          pairs:
            - BTCUSDT
            - ETHUSDT
          start: "2024-01-01"
          end: "2024-06-01"
          timeframe: 1h
          data_type: perpetual

        detector:
          name: example/sma_cross
          params:
            fast_period: 20
            slow_period: 50

        entry:
          size_pct: 0.1
          max_positions: 5
          max_per_pair: 1

        exit:
          tp: 0.03
          sl: 0.015
          trailing: 0.02

        capital: 50000
        fee: 0.001
        ```
    """

    strategy_id: str = "backtest"
    data: DataConfig | None = None
    detector: DetectorConfig | None = None
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
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
        # Parse nested configs
        data_cfg = None
        if "data" in d and d["data"]:
            data_cfg = DataConfig(**d["data"])

        detector_cfg = None
        if "detector" in d and d["detector"]:
            detector_cfg = DetectorConfig(
                name=d["detector"].get("name", ""),
                params=d["detector"].get("params", {}),
            )

        entry_cfg = EntryConfig()
        if "entry" in d and d["entry"]:
            entry_data = d["entry"]
            entry_cfg = EntryConfig(
                rule=entry_data.get("rule"),
                size=entry_data.get("size"),
                size_pct=entry_data.get("size_pct"),
                max_positions=entry_data.get("max_positions", 10),
                max_per_pair=entry_data.get("max_per_pair", 1),
            )

        exit_cfg = ExitConfig()
        if "exit" in d and d["exit"]:
            exit_data = d["exit"]
            exit_cfg = ExitConfig(
                rule=exit_data.get("rule"),
                tp=exit_data.get("tp"),
                sl=exit_data.get("sl"),
                trailing=exit_data.get("trailing"),
                time_limit=exit_data.get("time_limit"),
            )

        strategy_section = d.get("strategy", {})

        return cls(
            strategy_id=strategy_section.get("id", d.get("strategy_id", "backtest")),
            data=data_cfg,
            detector=detector_cfg,
            entry=entry_cfg,
            exit=exit_cfg,
            capital=d.get("capital", 10_000.0),
            fee=d.get("fee", 0.001),
            show_progress=d.get("show_progress", True),
        )

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of error/warning messages
        """
        issues: list[str] = []

        # Check data
        if self.data is None:
            issues.append("ERROR: No data source configured")
        else:
            if not self.data.source:
                issues.append("ERROR: data.source is required")
            if not self.data.pairs:
                issues.append("ERROR: data.pairs is required")
            if not self.data.start:
                issues.append("ERROR: data.start is required")

        # Check detector
        if self.detector is None:
            issues.append("ERROR: No detector configured")
        elif not self.detector.name:
            issues.append("ERROR: detector.name is required")

        # Validate TP/SL ratio
        if self.exit.tp and self.exit.sl:
            if self.exit.tp < self.exit.sl:
                issues.append(
                    f"WARNING: TP ({self.exit.tp:.1%}) < SL ({self.exit.sl:.1%}), "
                    "risk/reward < 1"
                )

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

        # Configure data
        if self.data:
            builder.data(
                exchange=None,  # Will use source path
                pairs=self.data.pairs,
                start=self.data.start,
                end=self.data.end,
                timeframe=self.data.timeframe,
                data_type=self.data.data_type,
            )
            # Override with source if it's a file path
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
                    pass  # Will be caught during run()

        # Configure detector
        if self.detector:
            builder.detector(self.detector.name, **self.detector.params)

        # Configure entry
        builder.entry(
            rule=self.entry.rule,
            size=self.entry.size,
            size_pct=self.entry.size_pct,
            max_positions=self.entry.max_positions,
            max_per_pair=self.entry.max_per_pair,
        )

        # Configure exit
        builder.exit(
            rule=self.exit.rule,
            tp=self.exit.tp,
            sl=self.exit.sl,
            trailing=self.exit.trailing,
            time_limit=self.exit.time_limit,
        )

        # Configure other
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
    return '''# SignalFlow Backtest Configuration
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
  timeframe: 1h                 # Candle timeframe
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
'''
