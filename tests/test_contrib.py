"""Tests for signalflow.contrib module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from signalflow.contrib.scaffold import scaffold, check_component, validate_component
from signalflow.core import SfComponentType


# ── Helpers ────────────────────────────────────────────────────────────


class _ValidDetector:
    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

    def detect(self, features: Any, context: Any = None) -> Any:
        return None


class _MissingMethod:
    component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR


class _NoComponentType:
    pass


class _WrongType:
    component_type = "not_an_enum"


class _ValidFeature:
    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE

    def compute_pair(self, df: Any) -> Any:
        return df


# ── scaffold ───────────────────────────────────────────────────────────


class TestScaffold:
    """Tests for scaffold() function."""

    @pytest.mark.parametrize("ct", ["detector", "feature", "validator", "labeler",
                                     "entry", "exit", "signal_feature"])
    def test_generates_file(self, tmp_path: Path, ct: str) -> None:
        path = scaffold("my_test_comp", component_type=ct, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".py"
        content = path.read_text()
        assert "class MyTestComp" in content
        assert "custom/my_test_comp" in content

    def test_detector_template_content(self, tmp_path: Path) -> None:
        path = scaffold("rsi_custom", component_type="detector", output_dir=tmp_path)
        content = path.read_text()
        assert "@sf.detector" in content
        assert "SignalDetector" in content
        assert "def detect" in content

    def test_feature_template_content(self, tmp_path: Path) -> None:
        path = scaffold("my_indicator", component_type="feature", output_dir=tmp_path)
        content = path.read_text()
        assert "@sf.feature" in content
        assert "def compute_pair" in content

    def test_unsupported_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            scaffold("foo", component_type="nonexistent", output_dir=tmp_path)

    def test_name_normalization(self, tmp_path: Path) -> None:
        path = scaffold("My-Custom Detector", component_type="detector", output_dir=tmp_path)
        assert path.name == "my_custom_detector.py"
        content = path.read_text()
        assert "class MyCustomDetector" in content


# ── validate_component ─────────────────────────────────────────────────


class TestValidateComponent:
    """Tests for validate_component() function."""

    def test_valid_detector(self) -> None:
        issues = validate_component(_ValidDetector)
        assert issues == []

    def test_valid_feature(self) -> None:
        issues = validate_component(_ValidFeature)
        assert issues == []

    def test_missing_component_type(self) -> None:
        issues = validate_component(_NoComponentType)
        assert any("component_type" in i for i in issues)

    def test_wrong_component_type(self) -> None:
        issues = validate_component(_WrongType)
        assert any("SfComponentType" in i for i in issues)

    def test_missing_required_method(self) -> None:
        issues = validate_component(_MissingMethod)
        assert any("detect" in i for i in issues)


# ── check_component ────────────────────────────────────────────────────


class TestCheckComponent:
    """Tests for check_component() function."""

    def test_valid_component(self) -> None:
        result = check_component(_ValidDetector)
        assert result["valid"] is True
        assert result["instance"] is not None
        assert result["issues"] == []

    def test_invalid_component(self) -> None:
        result = check_component(_NoComponentType)
        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_instantiation_failure(self) -> None:
        class _FailInit:
            component_type: ClassVar[SfComponentType] = SfComponentType.DETECTOR

            def __init__(self) -> None:
                raise RuntimeError("boom")

            def detect(self) -> None:
                pass

        result = check_component(_FailInit)
        assert result["valid"] is False
        assert any("Instantiation failed" in i for i in result["issues"])
