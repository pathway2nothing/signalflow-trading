"""Tests for signalflow.api.report module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from signalflow.api.report import Report, ReportSection, build_report

# ── Helpers ────────────────────────────────────────────────────────────


@dataclass
class _MockResult:
    """Minimal mock of BacktestResult."""

    strategy_id: str = "test_strategy"
    n_trades: int = 50
    total_return: float = 0.15
    win_rate: float = 0.60
    initial_capital: float = 10_000.0
    final_capital: float = 11_500.0
    metrics_df: Any = None
    config: dict[str, Any] | None = None
    _metrics: dict[str, float] | None = None
    _trades: list[dict[str, float]] | None = None

    @property
    def metrics(self) -> dict[str, float]:
        if self._metrics is not None:
            return self._metrics
        return {
            "sharpe_ratio": 1.85,
            "total_return": self.total_return,
            "max_drawdown": -0.08,
            "win_rate": self.win_rate,
            "profit_factor": 2.1,
            "n_trades": float(self.n_trades),
        }

    @property
    def trades(self) -> list[dict[str, float]]:
        if self._trades is not None:
            return self._trades
        return [
            {"pnl": 50.0},
            {"pnl": -20.0},
            {"pnl": 100.0},
            {"pnl": -15.0},
            {"pnl": 35.0},
        ]


# ── build_report ───────────────────────────────────────────────────────


class TestBuildReport:
    """Tests for build_report()."""

    def test_basic_report(self) -> None:
        result = _MockResult()
        report = build_report(result)
        assert isinstance(report, Report)
        assert report.strategy_id == "test_strategy"
        assert len(report.sections) > 0

    def test_has_summary_section(self) -> None:
        report = build_report(_MockResult())
        types = [s.content_type for s in report.sections]
        assert "summary" in types

    def test_has_metrics_section(self) -> None:
        report = build_report(_MockResult())
        types = [s.content_type for s in report.sections]
        assert "metrics_table" in types

    def test_has_trade_dist_section(self) -> None:
        report = build_report(_MockResult())
        types = [s.content_type for s in report.sections]
        assert "trade_dist" in types

    def test_has_config_section(self) -> None:
        result = _MockResult(config={"detector": "sma_cross", "capital": 10_000})
        report = build_report(result)
        types = [s.content_type for s in report.sections]
        assert "config" in types

    def test_no_config_when_empty(self) -> None:
        result = _MockResult(config=None)
        report = build_report(result)
        types = [s.content_type for s in report.sections]
        assert "config" not in types

    def test_no_trades(self) -> None:
        result = _MockResult(_trades=[])
        report = build_report(result)
        types = [s.content_type for s in report.sections]
        assert "trade_dist" not in types

    def test_summary_data_contains_key_fields(self) -> None:
        report = build_report(_MockResult())
        summary = next(s for s in report.sections if s.content_type == "summary")
        assert "trades" in summary.data
        assert "total_return" in summary.data
        assert "sharpe_ratio" in summary.data

    def test_trade_dist_values(self) -> None:
        result = _MockResult(
            _trades=[
                {"pnl": 100.0},
                {"pnl": -50.0},
                {"pnl": 200.0},
            ]
        )
        report = build_report(result)
        dist = next(s for s in report.sections if s.content_type == "trade_dist")
        assert dist.data["total_trades"] == 3
        assert dist.data["winners"] == 2
        assert dist.data["losers"] == 1

    def test_name_from_config(self) -> None:
        result = _MockResult(strategy_id="")
        result.config = {"strategy_id": "from_config"}
        report = build_report(result)
        assert report.strategy_id == "from_config"

    def test_fallback_name(self) -> None:
        result = _MockResult(strategy_id="")
        result.config = {}
        report = build_report(result)
        assert report.strategy_id == "Backtest Report"


# ── Report methods ─────────────────────────────────────────────────────


class TestReport:
    """Tests for Report class methods."""

    @pytest.fixture()
    def report(self) -> Report:
        return build_report(_MockResult(config={"detector": "sma_cross"}))

    def test_to_dict(self, report: Report) -> None:
        d = report.to_dict()
        assert "strategy_id" in d
        assert "sections" in d
        assert isinstance(d["sections"], list)
        for s in d["sections"]:
            assert "title" in s
            assert "content_type" in s
            assert "data" in s

    def test_text_summary(self, report: Report) -> None:
        text = report.text_summary()
        assert "test_strategy" in text
        assert "Summary" in text
        assert "Performance Metrics" in text

    def test_to_html_returns_string(self, report: Report) -> None:
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "test_strategy" in html
        assert "<table" in html

    def test_to_html_writes_file(self, report: Report, tmp_path: Any) -> None:
        path = tmp_path / "report.html"
        html = report.to_html(path)
        assert path.exists()
        content = path.read_text()
        assert content == html
        assert "<!DOCTYPE html>" in content

    def test_report_section_frozen(self) -> None:
        s = ReportSection(title="T", content_type="summary", data={})
        with pytest.raises(AttributeError):
            s.title = "changed"  # type: ignore[misc]
