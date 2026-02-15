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
                f"  from signalflow.core import sf_component, SfComponentType",
                "",
                f"  @sf_component(SfComponentType.{self.component_type.name}, 'my/{self.name}')",
                f"  class MyComponent:",
                f"      ...",
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


__all__ = [
    "SignalFlowError",
    "ConfigurationError",
    "DataError",
    "ComponentNotFoundError",
    "DetectorNotFoundError",
    "ValidatorNotFoundError",
    "DuplicateComponentNameError",
    "MissingDataError",
    "MissingDetectorError",
    "InvalidParameterError",
]
