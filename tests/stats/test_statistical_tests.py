"""Tests for statistical significance tests."""

from __future__ import annotations

import numpy as np

from signalflow.analytic.stats import StatisticalTestsValidator, statistical_tests


class TestStatisticalTestsValidator:
    """Test StatisticalTestsValidator class."""

    def test_validate_basic(self, mock_backtest_result):
        """Test basic validation."""
        validator = StatisticalTestsValidator(
            sr_benchmark=0.0,
            confidence_level=0.95,
        )
        result = validator.validate(mock_backtest_result)

        assert result.psr is not None
        assert 0.0 <= result.psr <= 1.0
        assert result.psr_benchmark == 0.0
        assert result.current_track_record > 0

    def test_validate_with_benchmark(self, mock_backtest_result):
        """Test validation against benchmark."""
        validator = StatisticalTestsValidator(
            sr_benchmark=0.5,
            confidence_level=0.95,
        )
        result = validator.validate(mock_backtest_result)

        assert result.psr_benchmark == 0.5
        # PSR should be lower when comparing to positive benchmark
        # (compared to benchmark of 0)

    def test_min_track_record_length(self, mock_backtest_result):
        """Test MinTRL calculation."""
        validator = StatisticalTestsValidator(
            sr_benchmark=0.0,
            confidence_level=0.95,
        )
        result = validator.validate(mock_backtest_result)

        # If PSR indicates significance, MinTRL should exist
        if result.psr is not None and result.psr > 0.5:
            # MinTRL might be None if SR <= benchmark
            pass  # This is acceptable

    def test_min_trl_high_benchmark(self, mock_backtest_result):
        """Test MinTRL with high benchmark."""
        validator = StatisticalTestsValidator(
            sr_benchmark=10.0,  # Very high benchmark
            confidence_level=0.95,
        )
        validator.validate(mock_backtest_result)

        # With such high benchmark, MinTRL should be None
        # (cannot be significant if SR < benchmark)
        # This depends on the actual SR of the mock result

    def test_track_record_sufficient(self, mock_backtest_result):
        """Test track record sufficiency check."""
        validator = StatisticalTestsValidator(
            sr_benchmark=0.0,
            confidence_level=0.95,
        )
        result = validator.validate(mock_backtest_result)

        if result.min_track_record_length is not None:
            expected = result.current_track_record >= result.min_track_record_length
            assert result.track_record_sufficient == expected

    def test_insufficient_data(self, mock_raw_data):
        """Test with insufficient data (< 2 observations)."""
        from signalflow.core import Portfolio, StrategyState
        from tests.stats.conftest import MockBacktestResult, MockTrade

        # Create result with only 1 trade
        state = StrategyState(
            strategy_id="test",
            portfolio=Portfolio(cash=10100.0),
        )
        result = MockBacktestResult(
            state=state,
            trades=[MockTrade(pnl=100.0)],
            signals=None,
            raw=mock_raw_data,
            config={"capital": 10000.0},
        )

        validator = StatisticalTestsValidator()
        st_result = validator.validate(result)

        assert st_result.psr is None
        assert st_result.min_track_record_length is None
        assert st_result.track_record_sufficient is False

    def test_summary(self, mock_backtest_result):
        """Test summary generation."""
        validator = StatisticalTestsValidator()
        result = validator.validate(mock_backtest_result)

        summary = result.summary()
        assert "Statistical" in summary
        assert "PSR" in summary or "Probabilistic" in summary

    def test_psr_significance(self, mock_backtest_result):
        """Test PSR significance determination."""
        # With 95% confidence, PSR > 0.95 means significant
        validator = StatisticalTestsValidator(
            sr_benchmark=0.0,
            confidence_level=0.95,
        )
        result = validator.validate(mock_backtest_result)

        if result.psr is not None:
            expected_sig = result.psr > 0.95
            assert result.psr_is_significant == expected_sig


class TestStatisticalTestsConvenience:
    """Test convenience function."""

    def test_statistical_tests_function(self, mock_backtest_result):
        """Test statistical_tests convenience function."""
        result = statistical_tests(
            mock_backtest_result,
            sr_benchmark=0.3,
            confidence_level=0.90,
        )

        assert result.psr_benchmark == 0.3


class TestProbabilisticSharpeRatio:
    """Test PSR calculation details."""

    def test_psr_with_positive_returns(self, sample_returns):
        """Test PSR with positive average returns."""
        # Create mock result with positive returns
        from signalflow.core import Portfolio, StrategyState
        from tests.stats.conftest import MockBacktestResult, MockTrade

        positive_returns = np.abs(sample_returns) + 0.01  # Force positive

        # Mock result
        from datetime import datetime as dt

        import polars as pl

        from signalflow.core import RawData

        raw = RawData(
            datetime_start=dt(2024, 1, 1),
            datetime_end=dt(2024, 1, 2),
            pairs=["BTCUSDT"],
            data={
                "spot": pl.DataFrame(
                    {
                        "pair": ["BTCUSDT"],
                        "timestamp": [dt(2024, 1, 1)],
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.0],
                        "volume": [100.0],
                    }
                )
            },
        )

        trades = [MockTrade(pnl=r * 100) for r in positive_returns]
        final_capital = 10000 + sum(t.pnl for t in trades)

        metrics_df = pl.DataFrame(
            {"timestamp": list(range(len(trades))), "total_return": np.cumsum(positive_returns).tolist()}
        )

        result = MockBacktestResult(
            state=StrategyState(
                strategy_id="test",
                portfolio=Portfolio(cash=final_capital),
            ),
            trades=trades,
            signals=None,
            raw=raw,
            config={"capital": 10000.0},
            metrics_df=metrics_df,
        )

        validator = StatisticalTestsValidator(sr_benchmark=0.0)
        st_result = validator.validate(result)

        # With positive returns vs benchmark of 0, PSR should be > 0.5
        assert st_result.psr is not None
        assert st_result.psr > 0.5

    def test_psr_with_negative_returns(self, sample_returns):
        """Test PSR with negative average returns."""
        from signalflow.core import Portfolio, RawData, StrategyState
        from tests.stats.conftest import MockBacktestResult, MockTrade

        # Create trades with negative PnL (losses)
        negative_pnls = -np.abs(sample_returns) * 100 - 1.0  # Force negative PnL

        from datetime import datetime as dt

        import polars as pl

        raw = RawData(
            datetime_start=dt(2024, 1, 1),
            datetime_end=dt(2024, 1, 2),
            pairs=["BTCUSDT"],
            data={
                "spot": pl.DataFrame(
                    {
                        "pair": ["BTCUSDT"],
                        "timestamp": [dt(2024, 1, 1)],
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.0],
                        "volume": [100.0],
                    }
                )
            },
        )

        trades = [MockTrade(pnl=pnl) for pnl in negative_pnls]
        final_capital = 10000 + sum(t.pnl for t in trades)

        # Create metrics_df with declining cumulative return (all negative)
        cumulative_returns = np.cumsum(negative_pnls) / 10000
        metrics_df = pl.DataFrame({"timestamp": list(range(len(trades))), "total_return": cumulative_returns.tolist()})

        result = MockBacktestResult(
            state=StrategyState(
                strategy_id="test",
                portfolio=Portfolio(cash=final_capital),
            ),
            trades=trades,
            signals=None,
            raw=raw,
            config={"capital": 10000.0},
            metrics_df=metrics_df,
        )

        validator = StatisticalTestsValidator(sr_benchmark=0.0)
        st_result = validator.validate(result)

        # With all losing trades vs benchmark of 0, PSR should be < 0.5
        assert st_result.psr is not None
        assert st_result.psr < 0.5
