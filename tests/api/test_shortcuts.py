"""Tests for signalflow.api.shortcuts module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from signalflow.api.shortcuts import _parse_datetime, backtest, load, load_artifact
from signalflow.core import RawData


class TestParseDateTime:
    """Tests for _parse_datetime helper."""

    def test_parse_none(self):
        """Returns None for None input."""
        assert _parse_datetime(None) is None

    def test_parse_datetime_passthrough(self):
        """Returns datetime as-is."""
        dt = datetime(2024, 1, 15, 10, 30)
        assert _parse_datetime(dt) is dt

    def test_parse_iso_string(self):
        """Parses ISO format string."""
        result = _parse_datetime("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_parse_iso_string_with_time(self):
        """Parses ISO format with time."""
        result = _parse_datetime("2024-01-15T10:30:00")
        assert result == datetime(2024, 1, 15, 10, 30)


class TestLoad:
    """Tests for load() function."""

    def test_load_requires_start(self, tmp_path):
        """Raises ValueError if start is not parseable."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()
        # None converts to None which is checked
        with pytest.raises(ValueError, match="start date is required"):
            load(db_path, pairs=["BTCUSDT"], start=None)  # type: ignore

    def test_load_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing DuckDB file."""
        with pytest.raises(FileNotFoundError, match="DuckDB file not found"):
            load(
                tmp_path / "missing.duckdb",
                pairs=["BTCUSDT"],
                start="2024-01-01",
            )

    def test_load_exchange_not_supported(self, tmp_path):
        """Raises ValueError for exchange names (not yet supported)."""
        with pytest.raises(ValueError, match="Direct exchange loading not yet supported"):
            load("binance", pairs=["BTCUSDT"], start="2024-01-01")

    def test_load_spot_data(self, tmp_path):
        """Load from spot DuckDB store calls correct factory method."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        mock_raw = MagicMock(spec=RawData)
        with patch(
            "signalflow.data.RawDataFactory.from_duckdb_spot_store",
            return_value=mock_raw,
        ) as mock_factory:
            result = load(
                db_path,
                pairs=["BTCUSDT"],
                start="2024-01-01",
                end="2024-02-01",
                timeframe="1h",
                data_type="spot",
            )

            mock_factory.assert_called_once()
            assert result is mock_raw

    def test_load_non_spot_data(self, tmp_path):
        """Load non-spot data uses StoreFactory."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        mock_store = MagicMock()
        mock_raw = MagicMock(spec=RawData)

        with (
            patch(
                "signalflow.data.StoreFactory.create_raw_store",
                return_value=mock_store,
            ) as mock_store_factory,
            patch(
                "signalflow.data.RawDataFactory.from_stores",
                return_value=mock_raw,
            ) as mock_factory,
        ):
            result = load(
                db_path,
                pairs=["BTCUSDT"],
                start="2024-01-01",
                data_type="perpetual",
            )

            mock_store_factory.assert_called_once()
            mock_factory.assert_called_once()
            assert result is mock_raw

    def test_load_with_path_object(self, tmp_path):
        """Accepts Path object as source."""
        db_path = tmp_path / "test.duckdb"
        db_path.touch()

        mock_raw = MagicMock(spec=RawData)
        with patch(
            "signalflow.data.RawDataFactory.from_duckdb_spot_store",
            return_value=mock_raw,
        ):
            result = load(
                db_path,  # Path object
                pairs=["BTCUSDT"],
                start=datetime(2024, 1, 1),
            )
            assert result is mock_raw


class TestLoadArtifact:
    """Tests for load_artifact() function."""

    def test_load_features(self, tmp_path):
        """Loads features.parquet artifact."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.write_parquet(tmp_path / "features.parquet")

        result = load_artifact(tmp_path, "features")
        assert result.height == 3
        assert "a" in result.columns

    def test_load_trades(self, tmp_path):
        """Loads trades.parquet artifact."""
        df = pl.DataFrame({"pnl": [100.0, -50.0, 200.0]})
        df.write_parquet(tmp_path / "trades.parquet")

        result = load_artifact(tmp_path, "trades")
        assert result.height == 3

    def test_load_artifact_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing artifact."""
        with pytest.raises(FileNotFoundError, match="Artifact not found"):
            load_artifact(tmp_path, "missing")

    def test_load_artifact_default_name(self, tmp_path):
        """Default artifact name is 'features'."""
        df = pl.DataFrame({"x": [1]})
        df.write_parquet(tmp_path / "features.parquet")

        result = load_artifact(tmp_path)  # No name specified
        assert result.height == 1

    def test_load_artifact_string_path(self, tmp_path):
        """Accepts string path."""
        df = pl.DataFrame({"x": [1]})
        df.write_parquet(tmp_path / "features.parquet")

        result = load_artifact(str(tmp_path), "features")
        assert result.height == 1


class TestBacktest:
    """Tests for backtest() function."""

    def test_backtest_requires_data(self):
        """Raises ValueError if no data provided."""
        from signalflow.detector import ExampleSmaCrossDetector

        detector = ExampleSmaCrossDetector()
        with pytest.raises(ValueError, match="'raw' or 'pairs'"):
            backtest(detector)

    def test_backtest_with_raw_data(self, sample_raw_data):
        """backtest() accepts RawData."""
        mock_result = MagicMock()

        with patch("signalflow.api.builder.BacktestBuilder.run", return_value=mock_result):
            result = backtest(
                "example/sma_cross",
                raw=sample_raw_data,
                tp=0.03,
                sl=0.015,
            )

        assert result is mock_result

    def test_backtest_with_detector_instance(self, sample_raw_data):
        """backtest() accepts detector instance."""
        from signalflow.detector import ExampleSmaCrossDetector

        detector = ExampleSmaCrossDetector(fast_period=10, slow_period=30)
        mock_result = MagicMock()

        with patch("signalflow.api.builder.BacktestBuilder.run", return_value=mock_result):
            result = backtest(detector, raw=sample_raw_data)

        assert result is mock_result

    def test_backtest_with_detector_string(self, sample_raw_data):
        """backtest() accepts detector registry name."""
        mock_result = MagicMock()

        with patch("signalflow.api.builder.BacktestBuilder.run", return_value=mock_result):
            result = backtest(
                "example/sma_cross",
                raw=sample_raw_data,
                fast_period=5,
                slow_period=20,
            )

        assert result is mock_result

    def test_backtest_configures_exit_rules(self, sample_raw_data):
        """backtest() configures TP/SL."""
        mock_result = MagicMock()

        with patch("signalflow.api.builder.BacktestBuilder") as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.data.return_value = mock_builder
            mock_builder.detector.return_value = mock_builder
            mock_builder.exit.return_value = mock_builder
            mock_builder.capital.return_value = mock_builder
            mock_builder.run.return_value = mock_result
            mock_builder_cls.return_value = mock_builder

            backtest(
                "example/sma_cross",
                raw=sample_raw_data,
                tp=0.05,
                sl=0.025,
                capital=50000.0,
            )

            mock_builder.exit.assert_called_once_with(tp=0.05, sl=0.025)
            mock_builder.capital.assert_called_once_with(50000.0)


@pytest.fixture
def sample_raw_data():
    """Create sample RawData for tests."""
    from datetime import timedelta

    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "pair": ["BTCUSDT"] * 100,
            "timestamp": [base + timedelta(hours=i) for i in range(100)],
            "open": [50000.0 + i * 10 for i in range(100)],
            "high": [50050.0 + i * 10 for i in range(100)],
            "low": [49950.0 + i * 10 for i in range(100)],
            "close": [50025.0 + i * 10 for i in range(100)],
            "volume": [1000.0] * 100,
        }
    )
    return RawData(
        datetime_start=base,
        datetime_end=base + timedelta(hours=100),
        pairs=["BTCUSDT"],
        data={"spot": df},
    )
