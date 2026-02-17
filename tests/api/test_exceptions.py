"""Tests for SignalFlow custom exceptions."""

import pytest

from signalflow.api.exceptions import (
    ColumnNotFoundError,
    ComponentNotFoundError,
    ConfigurationError,
    DataError,
    DetectorNotFoundError,
    DuplicateComponentNameError,
    InsufficientDataError,
    InvalidParameterError,
    LabelerNotFoundError,
    LookAheadBiasError,
    MissingDataError,
    MissingDetectorError,
    NoSignalsError,
    NoTradesError,
    SignalFlowError,
    ValidatorNotFoundError,
)
from signalflow.core import SfComponentType


# ===========================================================================
# Base Exception Tests
# ===========================================================================


class TestSignalFlowError:
    """Tests for base SignalFlowError."""

    def test_is_exception(self):
        """SignalFlowError is an Exception."""
        assert issubclass(SignalFlowError, Exception)

    def test_can_raise(self):
        """Can raise SignalFlowError."""
        with pytest.raises(SignalFlowError):
            raise SignalFlowError("test error")

    def test_message(self):
        """SignalFlowError stores message."""
        err = SignalFlowError("test message")
        assert str(err) == "test message"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inherits_signalflow_error(self):
        """ConfigurationError inherits from SignalFlowError."""
        assert issubclass(ConfigurationError, SignalFlowError)

    def test_can_raise(self):
        """Can raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("bad config")


class TestDataError:
    """Tests for DataError."""

    def test_inherits_signalflow_error(self):
        """DataError inherits from SignalFlowError."""
        assert issubclass(DataError, SignalFlowError)

    def test_can_raise(self):
        """Can raise DataError."""
        with pytest.raises(DataError):
            raise DataError("data problem")


# ===========================================================================
# ComponentNotFoundError Tests
# ===========================================================================


class TestComponentNotFoundError:
    """Tests for ComponentNotFoundError."""

    def test_inherits_signalflow_error(self):
        """ComponentNotFoundError inherits from SignalFlowError."""
        assert issubclass(ComponentNotFoundError, SignalFlowError)

    def test_stores_attributes(self):
        """Stores component_type, name, available."""
        err = ComponentNotFoundError(
            SfComponentType.DETECTOR,
            "my_detector",
            ["detector_a", "detector_b"],
        )
        assert err.component_type == SfComponentType.DETECTOR
        assert err.name == "my_detector"
        assert err.available == ["detector_a", "detector_b"]

    def test_message_includes_name(self):
        """Message includes component name."""
        err = ComponentNotFoundError(SfComponentType.DETECTOR, "my_detector")
        assert "my_detector" in str(err)

    def test_message_includes_type(self):
        """Message includes component type."""
        err = ComponentNotFoundError(SfComponentType.DETECTOR, "my_detector")
        assert "detector" in str(err).lower()

    def test_message_shows_available(self):
        """Message shows available options."""
        err = ComponentNotFoundError(
            SfComponentType.DETECTOR,
            "my_detector",
            ["detector_a", "detector_b"],
        )
        msg = str(err)
        assert "detector_a" in msg
        assert "detector_b" in msg

    def test_message_shows_truncated_available(self):
        """Message truncates long available list."""
        available = [f"detector_{i}" for i in range(15)]
        err = ComponentNotFoundError(SfComponentType.DETECTOR, "my_detector", available)
        msg = str(err)
        assert "... and 5 more" in msg

    def test_message_no_available(self):
        """Message handles empty available list."""
        err = ComponentNotFoundError(SfComponentType.DETECTOR, "my_detector", [])
        msg = str(err)
        assert "No components registered" in msg

    def test_message_includes_registration_hint(self):
        """Message includes registration hint."""
        err = ComponentNotFoundError(SfComponentType.DETECTOR, "my_detector")
        msg = str(err)
        assert "@sf_component" in msg


# ===========================================================================
# DetectorNotFoundError Tests
# ===========================================================================


class TestDetectorNotFoundError:
    """Tests for DetectorNotFoundError."""

    def test_inherits_component_not_found(self):
        """DetectorNotFoundError inherits from ComponentNotFoundError."""
        assert issubclass(DetectorNotFoundError, ComponentNotFoundError)

    def test_sets_component_type(self):
        """Sets component_type to DETECTOR."""
        err = DetectorNotFoundError("my_detector")
        assert err.component_type == SfComponentType.DETECTOR

    def test_message_includes_name(self):
        """Message includes detector name."""
        err = DetectorNotFoundError("my_detector")
        assert "my_detector" in str(err)

    def test_message_shows_available(self):
        """Message shows available detectors."""
        err = DetectorNotFoundError("my_detector", ["sma_cross", "rsi"])
        msg = str(err)
        assert "sma_cross" in msg
        assert "rsi" in msg

    def test_message_no_available(self):
        """Message suggests pip install when no detectors."""
        err = DetectorNotFoundError("my_detector", [])
        msg = str(err)
        assert "pip install signalflow-ta" in msg

    def test_message_includes_instance_hint(self):
        """Message includes hint about passing instance."""
        err = DetectorNotFoundError("my_detector")
        msg = str(err)
        assert "pass a detector instance" in msg.lower()


# ===========================================================================
# ValidatorNotFoundError Tests
# ===========================================================================


class TestValidatorNotFoundError:
    """Tests for ValidatorNotFoundError."""

    def test_inherits_component_not_found(self):
        """ValidatorNotFoundError inherits from ComponentNotFoundError."""
        assert issubclass(ValidatorNotFoundError, ComponentNotFoundError)

    def test_sets_component_type(self):
        """Sets component_type to VALIDATOR."""
        err = ValidatorNotFoundError("my_validator")
        assert err.component_type == SfComponentType.VALIDATOR

    def test_message_includes_name(self):
        """Message includes validator name."""
        err = ValidatorNotFoundError("my_validator")
        assert "my_validator" in str(err)

    def test_message_shows_available(self):
        """Message shows available validators."""
        err = ValidatorNotFoundError("my_validator", ["lgbm", "xgboost"])
        msg = str(err)
        assert "lgbm" in msg
        assert "xgboost" in msg

    def test_message_no_available(self):
        """Message handles no validators."""
        err = ValidatorNotFoundError("my_validator", [])
        msg = str(err)
        assert "No validators registered" in msg

    def test_message_includes_instance_hint(self):
        """Message includes hint about passing instance."""
        err = ValidatorNotFoundError("my_validator")
        msg = str(err)
        assert "pass a validator instance" in msg.lower()


# ===========================================================================
# LabelerNotFoundError Tests
# ===========================================================================


class TestLabelerNotFoundError:
    """Tests for LabelerNotFoundError."""

    def test_inherits_component_not_found(self):
        """LabelerNotFoundError inherits from ComponentNotFoundError."""
        assert issubclass(LabelerNotFoundError, ComponentNotFoundError)

    def test_sets_component_type(self):
        """Sets component_type to LABELER."""
        err = LabelerNotFoundError("my_labeler")
        assert err.component_type == SfComponentType.LABELER

    def test_message_includes_name(self):
        """Message includes labeler name."""
        err = LabelerNotFoundError("my_labeler")
        assert "my_labeler" in str(err)

    def test_message_shows_available(self):
        """Message shows available labelers."""
        err = LabelerNotFoundError("my_labeler", ["triple_barrier", "fixed_horizon"])
        msg = str(err)
        assert "triple_barrier" in msg
        assert "fixed_horizon" in msg

    def test_message_no_available(self):
        """Message shows built-in labelers when none registered."""
        err = LabelerNotFoundError("my_labeler", [])
        msg = str(err)
        assert "Built-in labelers" in msg
        assert "triple_barrier" in msg
        assert "fixed_horizon" in msg

    def test_message_includes_usage_example(self):
        """Message includes usage example."""
        err = LabelerNotFoundError("my_labeler")
        msg = str(err)
        assert ".labeler(" in msg


# ===========================================================================
# MissingDataError Tests
# ===========================================================================


class TestMissingDataError:
    """Tests for MissingDataError."""

    def test_inherits_configuration_error(self):
        """MissingDataError inherits from ConfigurationError."""
        assert issubclass(MissingDataError, ConfigurationError)

    def test_message_content(self):
        """Message explains how to configure data."""
        err = MissingDataError()
        msg = str(err)
        assert "No data configured" in msg
        assert ".data(" in msg
        assert "sf.load(" in msg


# ===========================================================================
# MissingDetectorError Tests
# ===========================================================================


class TestMissingDetectorError:
    """Tests for MissingDetectorError."""

    def test_inherits_configuration_error(self):
        """MissingDetectorError inherits from ConfigurationError."""
        assert issubclass(MissingDetectorError, ConfigurationError)

    def test_message_content(self):
        """Message explains how to configure detector."""
        err = MissingDetectorError()
        msg = str(err)
        assert "No detector or signals" in msg
        assert ".detector(" in msg
        assert ".signals(" in msg


# ===========================================================================
# InvalidParameterError Tests
# ===========================================================================


class TestInvalidParameterError:
    """Tests for InvalidParameterError."""

    def test_inherits_configuration_error(self):
        """InvalidParameterError inherits from ConfigurationError."""
        assert issubclass(InvalidParameterError, ConfigurationError)

    def test_stores_attributes(self):
        """Stores param, value, reason, hint."""
        err = InvalidParameterError(
            param="capital",
            value=-1000,
            reason="Must be positive",
            hint="Try capital=10000",
        )
        assert err.param == "capital"
        assert err.value == -1000
        assert err.reason == "Must be positive"
        assert err.hint == "Try capital=10000"

    def test_message_includes_param(self):
        """Message includes parameter name."""
        err = InvalidParameterError("capital", -1000, "Must be positive")
        assert "capital" in str(err)

    def test_message_includes_value(self):
        """Message includes invalid value."""
        err = InvalidParameterError("capital", -1000, "Must be positive")
        assert "-1000" in str(err)

    def test_message_includes_reason(self):
        """Message includes reason."""
        err = InvalidParameterError("capital", -1000, "Must be positive")
        assert "Must be positive" in str(err)

    def test_message_includes_hint(self):
        """Message includes hint if provided."""
        err = InvalidParameterError("capital", -1000, "Must be positive", "Try 10000")
        assert "Try 10000" in str(err)

    def test_message_without_hint(self):
        """Message works without hint."""
        err = InvalidParameterError("capital", -1000, "Must be positive")
        # Should not raise
        str(err)


# ===========================================================================
# DuplicateComponentNameError Tests
# ===========================================================================


class TestDuplicateComponentNameError:
    """Tests for DuplicateComponentNameError."""

    def test_inherits_configuration_error(self):
        """DuplicateComponentNameError inherits from ConfigurationError."""
        assert issubclass(DuplicateComponentNameError, ConfigurationError)

    def test_message_content(self):
        """Message includes component kind and name."""
        err = DuplicateComponentNameError("detector", "trend")
        msg = str(err)
        assert "Duplicate" in msg
        assert "detector" in msg
        assert "trend" in msg


# ===========================================================================
# InsufficientDataError Tests
# ===========================================================================


class TestInsufficientDataError:
    """Tests for InsufficientDataError."""

    def test_inherits_data_error(self):
        """InsufficientDataError inherits from DataError."""
        assert issubclass(InsufficientDataError, DataError)

    def test_stores_attributes(self):
        """Stores required and available."""
        err = InsufficientDataError(required=100, available=50)
        assert err.required == 100
        assert err.available == 50

    def test_message_includes_counts(self):
        """Message includes required and available counts."""
        err = InsufficientDataError(100, 50)
        msg = str(err)
        assert "100" in msg
        assert "50" in msg

    def test_message_includes_reason(self):
        """Message includes reason if provided."""
        err = InsufficientDataError(100, 50, reason="SMA needs 50 bars warmup")
        assert "SMA needs 50 bars warmup" in str(err)

    def test_message_includes_solutions(self):
        """Message includes possible solutions."""
        err = InsufficientDataError(100, 50)
        msg = str(err)
        assert "Load more historical data" in msg
        assert "Reduce indicator lookback" in msg


# ===========================================================================
# LookAheadBiasError Tests
# ===========================================================================


class TestLookAheadBiasError:
    """Tests for LookAheadBiasError."""

    def test_inherits_signalflow_error(self):
        """LookAheadBiasError inherits from SignalFlowError."""
        assert issubclass(LookAheadBiasError, SignalFlowError)

    def test_message_includes_component(self):
        """Message includes component name."""
        err = LookAheadBiasError("MyFeature", "Uses future price")
        assert "MyFeature" in str(err)

    def test_message_includes_detail(self):
        """Message includes detail."""
        err = LookAheadBiasError("MyFeature", "Uses future price")
        assert "Uses future price" in str(err)

    def test_message_includes_explanation(self):
        """Message explains look-ahead bias."""
        err = LookAheadBiasError("MyFeature", "Uses future price")
        msg = str(err)
        assert "future data" in msg.lower()
        assert "past decisions" in msg.lower()

    def test_message_includes_fix(self):
        """Message includes how to fix."""
        err = LookAheadBiasError("MyFeature", "Uses future price")
        msg = str(err)
        assert "How to fix" in msg
        assert "shift" in msg.lower()


# ===========================================================================
# NoSignalsError Tests
# ===========================================================================


class TestNoSignalsError:
    """Tests for NoSignalsError."""

    def test_inherits_signalflow_error(self):
        """NoSignalsError inherits from SignalFlowError."""
        assert issubclass(NoSignalsError, SignalFlowError)

    def test_message_includes_detector(self):
        """Message includes detector name."""
        err = NoSignalsError("RSI_Detector", 10000)
        assert "RSI_Detector" in str(err)

    def test_message_includes_row_count(self):
        """Message includes data row count."""
        err = NoSignalsError("RSI_Detector", 10000)
        assert "10,000" in str(err)

    def test_message_includes_causes(self):
        """Message includes possible causes."""
        err = NoSignalsError("RSI_Detector", 10000)
        msg = str(err)
        assert "Possible causes" in msg
        assert "parameters too strict" in msg.lower()

    def test_message_includes_suggestions(self):
        """Message includes suggestions."""
        err = NoSignalsError("RSI_Detector", 10000)
        msg = str(err)
        assert "Try" in msg
        assert "Relaxing" in msg


# ===========================================================================
# NoTradesError Tests
# ===========================================================================


class TestNoTradesError:
    """Tests for NoTradesError."""

    def test_inherits_signalflow_error(self):
        """NoTradesError inherits from SignalFlowError."""
        assert issubclass(NoTradesError, SignalFlowError)

    def test_message_includes_signal_count(self):
        """Message includes signal count."""
        err = NoTradesError(50)
        assert "50" in str(err)

    def test_message_includes_reason(self):
        """Message includes reason if provided."""
        err = NoTradesError(50, reason="All signals rejected by validator")
        assert "All signals rejected by validator" in str(err)

    def test_message_includes_causes(self):
        """Message includes possible causes."""
        err = NoTradesError(50)
        msg = str(err)
        assert "Possible causes" in msg
        assert "Validator rejected" in msg

    def test_message_includes_debug_tips(self):
        """Message includes debug tips."""
        err = NoTradesError(50)
        msg = str(err)
        assert "Debug tips" in msg
        assert ".validator(None)" in msg


# ===========================================================================
# ColumnNotFoundError Tests
# ===========================================================================


class TestColumnNotFoundError:
    """Tests for ColumnNotFoundError."""

    def test_inherits_data_error(self):
        """ColumnNotFoundError inherits from DataError."""
        assert issubclass(ColumnNotFoundError, DataError)

    def test_message_includes_column(self):
        """Message includes missing column name."""
        err = ColumnNotFoundError("rsi_14", ["open", "high", "low", "close"])
        assert "rsi_14" in str(err)

    def test_message_shows_available(self):
        """Message shows available columns."""
        err = ColumnNotFoundError("rsi_14", ["open", "high", "low", "close"])
        msg = str(err)
        assert "open" in msg
        assert "close" in msg

    def test_message_includes_context(self):
        """Message includes context if provided."""
        err = ColumnNotFoundError(
            "rsi_14",
            ["open", "close"],
            context="During feature extraction",
        )
        assert "During feature extraction" in str(err)

    def test_message_truncates_long_available(self):
        """Message truncates long available list."""
        available = [f"col_{i}" for i in range(20)]
        err = ColumnNotFoundError("rsi_14", available)
        msg = str(err)
        assert "... and 5 more" in msg

    def test_message_includes_causes(self):
        """Message includes common causes."""
        err = ColumnNotFoundError("rsi_14", ["open", "close"])
        msg = str(err)
        assert "Common causes" in msg
        assert "typo" in msg.lower()
