"""Lightweight monitoring alerts for realtime trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar

from loguru import logger

from signalflow.core.containers.signals import Signals
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.decorators import sf_component
from signalflow.core.enums import SfComponentType


@dataclass
class AlertResult:
    """Result from an alert check.

    Attributes:
        alert_name: Name of the alert that triggered.
        level: Alert severity - "warning" or "critical".
        message: Human-readable alert message.
        ts: Timestamp when alert was triggered.
        details: Additional structured data about the alert.
    """

    alert_name: str
    level: str
    message: str
    ts: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert(ABC):
    """Base class for monitoring alerts.

    Each alert checks a specific condition and returns an ``AlertResult``
    if the condition is met, or ``None`` otherwise.

    Attributes:
        enabled: Whether this alert is active.
    """

    enabled: bool = True

    @abstractmethod
    def check(
        self,
        state: StrategyState,
        signals: Signals,
        ts: datetime,
    ) -> AlertResult | None:
        """Check alert condition.

        Args:
            state: Current strategy state.
            signals: Signals for current bar.
            ts: Current timestamp.

        Returns:
            AlertResult if condition met, None otherwise.
        """
        ...


@dataclass
@sf_component(name="alert/max_drawdown")
class MaxDrawdownAlert(Alert):
    """Alert when drawdown exceeds threshold.

    Monitors the ``max_drawdown`` metric from strategy state and triggers
    warnings or critical alerts when thresholds are exceeded.

    Attributes:
        warning_threshold: Drawdown level for warning (e.g. 0.05 = 5%).
        critical_threshold: Drawdown level for critical alert (e.g. 0.10 = 10%).
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ALERT
    warning_threshold: float = 0.05
    critical_threshold: float = 0.10

    def check(
        self,
        state: StrategyState,
        signals: Signals,
        ts: datetime,
    ) -> AlertResult | None:
        drawdown = state.metrics.get("max_drawdown", 0.0)

        if drawdown >= self.critical_threshold:
            return AlertResult(
                alert_name="max_drawdown",
                level="critical",
                message=f"Max drawdown {drawdown:.2%} exceeds critical threshold {self.critical_threshold:.2%}",
                ts=ts,
                details={"drawdown": drawdown, "threshold": self.critical_threshold},
            )
        elif drawdown >= self.warning_threshold:
            return AlertResult(
                alert_name="max_drawdown",
                level="warning",
                message=f"Max drawdown {drawdown:.2%} exceeds warning threshold {self.warning_threshold:.2%}",
                ts=ts,
                details={"drawdown": drawdown, "threshold": self.warning_threshold},
            )

        return None


@dataclass
@sf_component(name="alert/no_signals")
class NoSignalsAlert(Alert):
    """Alert when no signals detected for too many bars.

    Tracks consecutive bars without any signals and triggers a warning
    after a configurable threshold.  Counter resets when signals are
    detected.

    Attributes:
        max_bars_without_signal: Number of consecutive bars without signals
            before alerting.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ALERT
    max_bars_without_signal: int = 100
    _bars_since_signal: int = field(default=0, init=False, repr=False)

    def check(
        self,
        state: StrategyState,
        signals: Signals,
        ts: datetime,
    ) -> AlertResult | None:
        if signals is not None and signals.value.height > 0:
            self._bars_since_signal = 0
            return None

        self._bars_since_signal += 1

        if self._bars_since_signal >= self.max_bars_without_signal:
            return AlertResult(
                alert_name="no_signals",
                level="warning",
                message=f"No signals detected for {self._bars_since_signal} consecutive bars",
                ts=ts,
                details={"bars_since_signal": self._bars_since_signal},
            )

        return None


@dataclass
@sf_component(name="alert/stuck_position")
class StuckPositionAlert(Alert):
    """Alert when positions have been open too long.

    Monitors position ages and triggers warnings when any position exceeds
    the maximum allowed duration.

    Attributes:
        max_duration: Maximum duration a position should be open.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ALERT
    max_duration: timedelta = field(default_factory=lambda: timedelta(days=7))

    def check(
        self,
        state: StrategyState,
        signals: Signals,
        ts: datetime,
    ) -> AlertResult | None:
        if not hasattr(state.portfolio, "open_positions"):
            return None

        for position in state.portfolio.open_positions():
            if position.entry_time is None:
                continue

            age = ts - position.entry_time
            if age > self.max_duration:
                return AlertResult(
                    alert_name="stuck_position",
                    level="warning",
                    message=(
                        f"Position {position.id[:8]} on {position.pair} "
                        f"has been open for {age} (max={self.max_duration})"
                    ),
                    ts=ts,
                    details={
                        "position_id": position.id,
                        "pair": position.pair,
                        "age": str(age),
                        "entry_price": position.entry_price,
                        "current_price": position.last_price,
                    },
                )

        return None


@dataclass
class AlertManager:
    """Manages a collection of alerts.

    Runs all alerts, logs triggered results, and maintains a history
    of all alerts for post-analysis.

    Attributes:
        alerts: List of Alert instances to check.

    Example::

        manager = AlertManager(alerts=[
            MaxDrawdownAlert(warning_threshold=0.05, critical_threshold=0.10),
            NoSignalsAlert(max_bars_without_signal=50),
            StuckPositionAlert(max_duration=timedelta(days=3)),
        ])

        # In the runner loop:
        results = manager.check_all(state, signals, ts)
    """

    alerts: list[Alert] = field(default_factory=list)
    _alert_history: list[AlertResult] = field(default_factory=list, init=False, repr=False)

    def check_all(
        self,
        state: StrategyState,
        signals: Signals,
        ts: datetime,
    ) -> list[AlertResult]:
        """Run all alert checks and log any triggered alerts.

        Args:
            state: Current strategy state.
            signals: Signals for current bar.
            ts: Current timestamp.

        Returns:
            List of triggered AlertResults.
        """
        results: list[AlertResult] = []

        for alert in self.alerts:
            if not alert.enabled:
                continue

            try:
                result = alert.check(state, signals, ts)
            except Exception:
                logger.exception(f"Alert check failed: {alert.__class__.__name__}")
                continue

            if result is not None:
                results.append(result)
                self._alert_history.append(result)

                if result.level == "critical":
                    logger.critical(
                        f"ALERT [{result.alert_name}] {result.message}",
                    )
                else:
                    logger.warning(
                        f"ALERT [{result.alert_name}] {result.message}",
                    )

        return results

    @property
    def alert_history(self) -> list[AlertResult]:
        """All alerts triggered during this session."""
        return self._alert_history
