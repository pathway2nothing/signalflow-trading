"""
Custom exceptions for SignalFlow API.

These provide clear, actionable error messages for common issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signalflow.core import SfComponentType


class SignalFlowError(Exception):
    """Base exception for SignalFlow errors."""

    pass


class ConfigurationError(SignalFlowError):
    """Raised when backtest configuration is invalid."""

    pass


class DataError(SignalFlowError):
    """Raised when there's an issue with data loading or access."""

    pass


class ComponentNotFoundError(SignalFlowError):
    """Raised when a registry component is not found."""

    def __init__(
        self,
        component_type: SfComponentType,
        name: str,
        available: list[str] | None = None,
    ):
        self.component_type = component_type
        self.name = name
        self.available = available or []

        msg = self._build_message()
        super().__init__(msg)

    def _build_message(self) -> str:
        type_name = self.component_type.name.lower().replace("_", " ")

        lines = [
            f"'{self.name}' not found in {type_name} registry.",
            "",
        ]

        if self.available:
            lines.append("Available options:")
            for opt in sorted(self.available)[:10]:
                lines.append(f"  - {opt}")
            if len(self.available) > 10:
                lines.append(f"  ... and {len(self.available) - 10} more")
        else:
            lines.append("No components registered for this type.")

        lines.extend(
            [
                "",
                "To register a custom component, use @sf_component decorator:",
                "",
                "  from signalflow.core import sf_component, SfComponentType",
                "",
                f"  @sf_component(SfComponentType.{self.component_type.name}, 'my/{self.name}')",
                "  class MyComponent:",
                "      ...",
            ]
        )

        return "\n".join(lines)


class DetectorNotFoundError(ComponentNotFoundError):
    """Raised when a detector is not found in registry."""

    def __init__(self, name: str, available: list[str] | None = None):
        from signalflow.core import SfComponentType

        super().__init__(SfComponentType.DETECTOR, name, available)

    def _build_message(self) -> str:
        lines = [
            f"Detector '{self.name}' not found in registry.",
            "",
        ]

        if self.available:
            lines.append("Available detectors:")
            for opt in sorted(self.available)[:10]:
                lines.append(f"  - {opt}")
            if len(self.available) > 10:
                lines.append(f"  ... and {len(self.available) - 10} more")
        else:
            lines.append("No detectors registered. Install signalflow-ta for built-in detectors:")
            lines.append("  pip install signalflow-ta")

        lines.extend(
            [
                "",
                "Or pass a detector instance directly:",
                "",
                "  from signalflow.detector import ExampleSmaCrossDetector",
                "",
                "  sf.Backtest()",
                "      .detector(ExampleSmaCrossDetector(fast_period=20, slow_period=50))",
                "      ...",
            ]
        )

        return "\n".join(lines)


class MissingDataError(ConfigurationError):
    """Raised when data is not configured for backtest."""

    def __init__(self):
        msg = "\n".join(
            [
                "No data configured for backtest.",
                "",
                "Configure data using one of these methods:",
                "",
                "  # Option 1: Pre-loaded RawData",
                "  sf.Backtest()",
                "      .data(raw=my_raw_data)",
                "      ...",
                "",
                "  # Option 2: Load from DuckDB file",
                "  raw = sf.load('data/binance.duckdb', pairs=['BTCUSDT'], start='2024-01-01')",
                "  sf.Backtest()",
                "      .data(raw=raw)",
                "      ...",
            ]
        )
        super().__init__(msg)


class MissingDetectorError(ConfigurationError):
    """Raised when detector is not configured for backtest."""

    def __init__(self):
        msg = "\n".join(
            [
                "No detector or signals configured for backtest.",
                "",
                "Configure using one of these methods:",
                "",
                "  # Option 1: Registry name (requires signalflow-ta or custom registration)",
                "  sf.Backtest()",
                "      .detector('namespace/detector_name', param=value)",
                "      ...",
                "",
                "  # Option 2: Detector instance",
                "  from signalflow.detector import ExampleSmaCrossDetector",
                "  detector = ExampleSmaCrossDetector(fast_period=20, slow_period=50)",
                "  sf.Backtest()",
                "      .detector(detector)",
                "      ...",
                "",
                "  # Option 3: Pre-computed signals",
                "  sf.Backtest()",
                "      .signals(my_signals)",
                "      ...",
            ]
        )
        super().__init__(msg)


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter value is invalid."""

    def __init__(self, param: str, value: object, reason: str, hint: str | None = None):
        self.param = param
        self.value = value
        self.reason = reason
        self.hint = hint

        lines = [
            f"Invalid value for '{param}': {value!r}",
            f"  {reason}",
        ]
        if hint:
            lines.extend(["", f"Hint: {hint}"])

        super().__init__("\n".join(lines))


class ValidatorNotFoundError(ComponentNotFoundError):
    """Raised when a validator is not found in registry."""

    def __init__(self, name: str, available: list[str] | None = None):
        from signalflow.core import SfComponentType

        super().__init__(SfComponentType.VALIDATOR, name, available)

    def _build_message(self) -> str:
        lines = [
            f"Validator '{self.name}' not found in registry.",
            "",
        ]

        if self.available:
            lines.append("Available validators:")
            for opt in sorted(self.available)[:10]:
                lines.append(f"  - {opt}")
        else:
            lines.append("No validators registered.")

        lines.extend(
            [
                "",
                "Or pass a validator instance directly:",
                "",
                "  from signalflow.validator import LightGBMValidator",
                "",
                "  sf.Backtest()",
                "      .validator(LightGBMValidator())",
                "      ...",
            ]
        )

        return "\n".join(lines)


class DuplicateComponentNameError(ConfigurationError):
    """Raised when a component name is already registered."""

    def __init__(self, component_kind: str, name: str):
        msg = f"Duplicate {component_kind} name: '{name}'. Each {component_kind} must have a unique name."
        super().__init__(msg)


class LabelerNotFoundError(ComponentNotFoundError):
    """Raised when a labeler is not found in registry."""

    def __init__(self, name: str, available: list[str] | None = None):
        from signalflow.core import SfComponentType

        super().__init__(SfComponentType.LABELER, name, available)

    def _build_message(self) -> str:
        lines = [
            f"Labeler '{self.name}' not found in registry.",
            "",
        ]

        if self.available:
            lines.append("Available labelers:")
            for opt in sorted(self.available)[:10]:
                lines.append(f"  - {opt}")
        else:
            lines.append("Built-in labelers:")
            lines.append("  - triple_barrier  (take-profit/stop-loss/time barriers)")
            lines.append("  - fixed_horizon   (label based on future return)")

        lines.extend(
            [
                "",
                "Example usage:",
                "",
                "  sf.Backtest()",
                "      .labeler('triple_barrier', tp=0.02, sl=0.01, horizon=20)",
                "      ...",
            ]
        )

        return "\n".join(lines)


class InsufficientDataError(DataError):
    """Raised when there's not enough data for the requested operation."""

    def __init__(
        self,
        required: int,
        available: int,
        reason: str | None = None,
    ):
        self.required = required
        self.available = available

        lines = [
            f"Insufficient data: need at least {required} bars, but only {available} available.",
        ]

        if reason:
            lines.append(f"  Reason: {reason}")

        lines.extend(
            [
                "",
                "Possible solutions:",
                "  1. Load more historical data",
                "  2. Reduce indicator lookback periods",
                "  3. Use a shorter timeframe (e.g., '1h' instead of '4h')",
            ]
        )

        super().__init__("\n".join(lines))


class LookAheadBiasError(SignalFlowError):
    """Raised when look-ahead bias is detected in the pipeline."""

    def __init__(self, component: str, detail: str):
        lines = [
            f"Look-ahead bias detected in '{component}'.",
            f"  {detail}",
            "",
            "Look-ahead bias means using future data to make past decisions.",
            "This invalidates backtest results.",
            "",
            "How to fix:",
            "  - Ensure all features use only past data (shift by 1 or more)",
            "  - Check that labels don't leak into features",
            "  - Verify train/test split respects time order",
        ]
        super().__init__("\n".join(lines))


class NoSignalsError(SignalFlowError):
    """Raised when detector produces no signals."""

    def __init__(self, detector_name: str, data_rows: int):
        lines = [
            f"Detector '{detector_name}' produced 0 signals from {data_rows:,} rows.",
            "",
            "Possible causes:",
            "  1. Detector parameters too strict (e.g., RSI threshold never reached)",
            "  2. Data range too short for indicator warmup period",
            "  3. Market conditions don't match detector logic",
            "",
            "Try:",
            "  - Relaxing detector parameters",
            "  - Using more historical data",
            "  - Checking detector logic with .detector_result() debug output",
        ]
        super().__init__("\n".join(lines))


class NoTradesError(SignalFlowError):
    """Raised when backtest produces no trades."""

    def __init__(self, signals_count: int, reason: str | None = None):
        lines = [
            f"Backtest produced 0 trades from {signals_count} signals.",
        ]

        if reason:
            lines.append(f"  Reason: {reason}")

        lines.extend(
            [
                "",
                "Possible causes:",
                "  1. Validator rejected all signals (check validation threshold)",
                "  2. Entry filters blocked all entries",
                "  3. Exit rules prevented trade completion",
                "",
                "Debug tips:",
                "  - Run without validator: .validator(None)",
                "  - Check validator predictions: result.validation_result",
                "  - Lower validation threshold: .validator('lgbm', threshold=0.3)",
            ]
        )
        super().__init__("\n".join(lines))


class ColumnNotFoundError(DataError):
    """Raised when a required column is missing from data."""

    def __init__(self, column: str, available: list[str], context: str | None = None):
        lines = [
            f"Required column '{column}' not found in data.",
        ]

        if context:
            lines.append(f"  Context: {context}")

        lines.append("")
        lines.append("Available columns:")
        for col in sorted(available)[:15]:
            lines.append(f"  - {col}")
        if len(available) > 15:
            lines.append(f"  ... and {len(available) - 15} more")

        lines.extend(
            [
                "",
                "Common causes:",
                "  - Column name typo",
                "  - Feature pipeline not applied before validator",
                "  - Data schema mismatch between train/test",
            ]
        )

        super().__init__("\n".join(lines))


__all__ = [
    # Base
    "SignalFlowError",
    "ConfigurationError",
    "DataError",
    # Component errors
    "ComponentNotFoundError",
    "DetectorNotFoundError",
    "ValidatorNotFoundError",
    "LabelerNotFoundError",
    "DuplicateComponentNameError",
    # Configuration errors
    "MissingDataError",
    "MissingDetectorError",
    "InvalidParameterError",
    # Data errors
    "InsufficientDataError",
    "ColumnNotFoundError",
    # Runtime errors
    "LookAheadBiasError",
    "NoSignalsError",
    "NoTradesError",
]
