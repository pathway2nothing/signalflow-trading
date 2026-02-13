"""Tests for BacktestResult."""

import pytest

from signalflow.api.result import BacktestResult
from signalflow.core import default_registry, SfComponentType


class TestBacktestResultProperties:
    """Tests for BacktestResult basic properties."""

    def test_n_trades(self, sample_result):
        """n_trades returns correct count."""
        assert sample_result.n_trades == 5

    def test_n_trades_empty(self, sample_state, sample_raw_data):
        """n_trades returns 0 for empty trades."""
        result = BacktestResult(
            state=sample_state,
            trades=[],
            signals=None,
            raw=sample_raw_data,
            config={"capital": 10_000.0},
        )
        assert result.n_trades == 0

    def test_final_capital(self, sample_result):
        """final_capital returns state.capital."""
        assert sample_result.final_capital == 12000.0

    def test_initial_capital_from_config(self, sample_result):
        """initial_capital comes from config."""
        assert sample_result.initial_capital == 10_000.0

    def test_total_return_calculation(self, sample_result):
        """total_return is (final - initial) / initial."""
        # (12000 - 10000) / 10000 = 0.2
        assert sample_result.total_return == pytest.approx(0.2)

    def test_total_return_pct(self, sample_result):
        """total_return_pct is total_return * 100."""
        assert sample_result.total_return_pct == pytest.approx(20.0)

    def test_win_rate_calculation(self, sample_result):
        """win_rate is wins / total trades."""
        # 3 wins out of 5
        assert sample_result.win_rate == pytest.approx(0.6)

    def test_win_rate_empty_trades(self, sample_state, sample_raw_data):
        """win_rate returns 0 for empty trades."""
        result = BacktestResult(
            state=sample_state,
            trades=[],
            signals=None,
            raw=sample_raw_data,
            config={"capital": 10_000.0},
        )
        assert result.win_rate == 0.0

    def test_profit_factor(self, sample_result):
        """profit_factor is gross_profit / gross_loss."""
        # Profits: 100 + 200 + 150 = 450
        # Losses: 50 + 30 = 80
        # PF = 450 / 80 = 5.625
        assert sample_result.profit_factor == pytest.approx(5.625)


class TestBacktestResultMetrics:
    """Tests for metrics computation."""

    def test_metrics_returns_dict(self, sample_result):
        """metrics property returns dict."""
        metrics = sample_result.metrics
        assert isinstance(metrics, dict)

    def test_metrics_includes_basic(self, sample_result):
        """metrics includes basic calculated values."""
        metrics = sample_result.metrics
        assert "n_trades" in metrics
        assert "win_rate" in metrics
        assert "total_return" in metrics
        assert "profit_factor" in metrics

    def test_metrics_cached(self, sample_result):
        """metrics are cached after first access."""
        metrics1 = sample_result.metrics
        metrics2 = sample_result.metrics
        assert metrics1 is metrics2


class TestBacktestResultSummary:
    """Tests for summary output."""

    def test_summary_returns_string(self, sample_result):
        """summary() returns formatted string."""
        summary = sample_result.summary()
        assert isinstance(summary, str)

    def test_summary_contains_key_info(self, sample_result):
        """summary() contains key metrics."""
        summary = sample_result.summary()
        assert "Trades:" in summary
        assert "Win Rate:" in summary
        assert "Total Return:" in summary
        assert "Final Capital:" in summary

    def test_repr(self, sample_result):
        """__repr__ is readable."""
        repr_str = repr(sample_result)
        assert "BacktestResult" in repr_str
        assert "trades=" in repr_str
        assert "return=" in repr_str


class TestBacktestResultExport:
    """Tests for export functionality."""

    def test_to_dict(self, sample_result):
        """to_dict() exports as dictionary."""
        d = sample_result.to_dict()
        assert isinstance(d, dict)
        assert "metrics" in d
        assert "n_trades" in d
        assert "trades" in d
        assert "config" in d


class TestBacktestResultVisualization:
    """Tests for visualization methods."""

    def test_plot_uses_registry(self, sample_result):
        """plot() uses StrategyMainResult from registry."""
        # Check registry has the metric
        metrics = default_registry.list(SfComponentType.STRATEGY_METRIC)
        if "result_main" in metrics:
            # Should not crash
            figs = sample_result.plot()
            # May return None if no metrics_df
            assert figs is None or isinstance(figs, list)

    def test_plot_pair_uses_registry(self, sample_result):
        """plot_pair() uses StrategyPairResult from registry."""
        metrics = default_registry.list(SfComponentType.STRATEGY_METRIC)
        if "result_pair" in metrics:
            figs = sample_result.plot_pair("BTCUSDT")
            assert figs is None or isinstance(figs, list)


class TestBacktestResultJupyter:
    """Tests for Jupyter notebook support."""

    def test_repr_html_returns_string(self, sample_result):
        """_repr_html_() returns HTML string."""
        html = sample_result._repr_html_()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_repr_html_contains_metrics(self, sample_result):
        """_repr_html_() contains key metrics."""
        html = sample_result._repr_html_()
        assert "BACKTEST RESULT" in html
        assert "Trades" in html
        assert "Win Rate" in html
        assert "Profit Factor" in html

    def test_repr_html_shows_return(self, sample_result):
        """_repr_html_() shows total return prominently."""
        html = sample_result._repr_html_()
        # Should contain the return value
        assert "%" in html

    def test_repr_html_has_styling(self, sample_result):
        """_repr_html_() has CSS styling."""
        html = sample_result._repr_html_()
        assert "style=" in html
        assert "font-family" in html

    def test_to_dataframe_returns_polars(self, sample_result):
        """to_dataframe() returns Polars DataFrame."""
        import polars as pl

        df = sample_result.to_dataframe()
        assert isinstance(df, pl.DataFrame)

    def test_to_dataframe_empty_trades(self, sample_state, sample_raw_data, sample_signals):
        """to_dataframe() handles empty trades."""
        import polars as pl
        from signalflow.api.result import BacktestResult

        result = BacktestResult(
            state=sample_state,
            trades=[],
            signals=sample_signals,
            raw=sample_raw_data,
            config={"capital": 10_000.0},
        )
        df = result.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
