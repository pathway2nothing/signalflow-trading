"""Tests for signalflow._help module (sf.help() API)."""

from __future__ import annotations

import pytest

from signalflow._help import _search, help_system
from signalflow.help_glossary import GLOSSARY


class TestHelpSystem:
    """Tests for the _HelpSystem callable."""

    def test_is_callable(self) -> None:
        assert callable(help_system)

    def test_overview_no_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system()
        out = capsys.readouterr().out
        assert "SignalFlow Help" in out
        assert "Component Types:" in out

    def test_glossary_term_lookup(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system("sharpe_ratio")
        out = capsys.readouterr().out
        assert "sharpe_ratio" in out
        assert "Risk-adjusted" in out
        assert "Formula:" in out

    def test_glossary_case_insensitive(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system("Sharpe_Ratio")
        out = capsys.readouterr().out
        assert "sharpe_ratio" in out

    def test_type_alias_lookup(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system("detectors")
        out = capsys.readouterr().out
        assert "signals/detector" in out
        assert "registered" in out

    def test_component_exact_lookup(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system("example/sma_cross")
        out = capsys.readouterr().out
        assert "sma_cross" in out
        assert "signals/detector" in out

    def test_component_suffix_lookup(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Partial name like 'sma_cross' should match 'example/sma_cross'."""
        help_system("sma_cross")
        out = capsys.readouterr().out
        assert "example/sma_cross" in out

    def test_unknown_term_falls_back_to_search(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system("xyznonexistent123")
        out = capsys.readouterr().out
        assert "No results" in out

    def test_search_method(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system.search("momentum")
        out = capsys.readouterr().out
        assert "momentum" in out.lower()
        assert "found" in out

    def test_detectors_method(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system.detectors()
        out = capsys.readouterr().out
        assert "signals/detector" in out

    def test_metrics_method(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system.metrics()
        out = capsys.readouterr().out
        assert "strategy/metric" in out

    def test_glossary_method(self, capsys: pytest.CaptureFixture[str]) -> None:
        help_system.glossary()
        out = capsys.readouterr().out
        assert "Glossary" in out
        assert "sharpe_ratio" in out

    def test_schema_method(self) -> None:
        schema = help_system.schema("detector", "example/sma_cross")
        assert "name" in schema
        assert "parameters" in schema

    def test_schema_invalid_type_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown component type"):
            help_system.schema("nonexistent_type", "foo")

    def test_export_all(self) -> None:
        export = help_system.export_all()
        assert "components" in export
        assert "glossary" in export
        assert len(export["glossary"]) == len(GLOSSARY)


class TestSearch:
    """Tests for _search() function."""

    def test_finds_glossary_matches(self) -> None:
        result = _search("sharpe")
        assert "[glossary]" in result
        assert "sharpe_ratio" in result

    def test_finds_component_matches(self) -> None:
        result = _search("sma")
        assert "sma_cross" in result

    def test_no_results(self) -> None:
        result = _search("zzzznonexistent999")
        assert "No results" in result


class TestGlossary:
    """Tests for glossary data integrity."""

    def test_glossary_not_empty(self) -> None:
        assert len(GLOSSARY) > 30

    def test_all_entries_have_definition(self) -> None:
        for term, entry in GLOSSARY.items():
            assert "definition" in entry, f"{term} missing 'definition'"

    def test_all_entries_have_category(self) -> None:
        for term, entry in GLOSSARY.items():
            assert "category" in entry, f"{term} missing 'category'"

    def test_key_terms_present(self) -> None:
        expected = [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "rsi",
            "macd",
            "bollinger_bands",
            "monte_carlo",
            "bootstrap",
            "psr",
        ]
        for term in expected:
            assert term in GLOSSARY, f"Missing glossary term: {term}"
