"""Tests for FlowBuilder and FlowResult."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from signalflow.api.flow import (
    AggregationMode,
    BacktestMetrics,
    FeatureMetrics,
    FlowBuilder,
    FlowConfig,
    FlowResult,
    LabelMetrics,
    LiveMetrics,
    RunMode,
    SignalMetrics,
    ValidationMetrics,
    flow,
)
from signalflow.core import RawData, Signals


# ===========================================================================
# FlowResult Tests
# ===========================================================================


class TestFlowResult:
    """Tests for FlowResult dataclass."""

    def test_default_values(self):
        """FlowResult has correct defaults."""
        result = FlowResult()
        assert result.feature_metrics is None
        assert result.signal_metrics is None
        assert result.label_metrics is None
        assert result.validation_metrics is None
        assert result.backtest_metrics is None
        assert result.live_metrics is None
        assert result.features is None
        assert result.signals is None
        assert result.labels is None
        assert result.predictions is None
        assert result.trades is None
        assert result.flow_config is None
        assert result.execution_time == 0.0
        assert result.fold_results is None
        assert result.window_results is None

    def test_summary_empty(self):
        """summary() works with no metrics."""
        result = FlowResult()
        summary = result.summary()
        assert "FLOW RESULT SUMMARY" in summary
        assert "Execution time: 0.00s" in summary

    def test_summary_with_backtest_metrics(self):
        """summary() includes backtest metrics."""
        result = FlowResult()
        result.backtest_metrics = {"sharpe_ratio": 1.5, "total_return": 0.25}
        summary = result.summary()
        assert "Backtest Metrics" in summary
        assert "sharpe_ratio" in summary
        assert "total_return" in summary

    def test_summary_with_signal_metrics(self):
        """summary() includes signal metrics."""
        result = FlowResult()
        result.signal_metrics = {"signal_rate": 0.05, "active_signals": 100}
        summary = result.summary()
        assert "Signal Metrics" in summary
        assert "signal_rate" in summary

    def test_summary_with_feature_metrics(self):
        """summary() includes feature metrics."""
        result = FlowResult()
        result.feature_metrics = {"n_features": 20, "null_pct": 0.01}
        summary = result.summary()
        assert "Feature Metrics" in summary
        assert "n_features" in summary

    def test_summary_with_validation_metrics(self):
        """summary() includes validation metrics."""
        result = FlowResult()
        result.validation_metrics = {"accuracy": 0.8}
        summary = result.summary()
        assert "Validation Metrics" in summary

    def test_format_metrics_dict(self):
        """_format_metrics handles dict."""
        result = FlowResult()
        formatted = result._format_metrics({"a": 0.1234, "b": 12345.0})
        # Small floats get 4 decimals
        assert formatted["a"] == "0.1234"
        # Large floats get 2 decimals with commas
        assert formatted["b"] == "12,345.00"

    def test_format_metrics_object(self):
        """_format_metrics handles object with __dict__."""
        result = FlowResult()

        class Metrics:
            def __init__(self):
                self.sharpe = 1.5
                self._private = "hidden"

        formatted = result._format_metrics(Metrics())
        assert "sharpe" in formatted
        assert "_private" not in formatted

    def test_format_value_float_small(self):
        """_format_value formats small floats with 4 decimals."""
        assert FlowResult._format_value(0.1234) == "0.1234"

    def test_format_value_float_large(self):
        """_format_value formats large floats with 2 decimals and commas."""
        assert FlowResult._format_value(12345.67) == "12,345.67"

    def test_format_value_string(self):
        """_format_value converts non-float to string."""
        assert FlowResult._format_value("test") == "test"
        assert FlowResult._format_value(123) == "123"

    def test_to_json_dict_empty(self):
        """to_json_dict() works with empty result."""
        result = FlowResult()
        json_dict = result.to_json_dict()
        assert json_dict["n_trades"] == 0
        assert json_dict["execution_time"] == 0.0

    def test_to_json_dict_with_metrics(self):
        """to_json_dict() includes backtest metrics."""
        result = FlowResult()
        result.backtest_metrics = {"sharpe": 1.5}
        result.execution_time = 5.0
        json_dict = result.to_json_dict()
        assert json_dict["metrics"] == {"sharpe": 1.5}
        assert json_dict["execution_time"] == 5.0

    def test_to_json_dict_with_trades(self):
        """to_json_dict() includes trades if present."""
        result = FlowResult()
        result.trades = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "pnl": [100.0],
            }
        )
        json_dict = result.to_json_dict()
        assert json_dict["n_trades"] == 1
        assert "trades" in json_dict

    def test_to_json_dict_with_config(self):
        """to_json_dict() includes config."""
        result = FlowResult()
        result.flow_config = FlowConfig(
            strategy_id="test",
            run_mode=RunMode.SINGLE,
            data_sources=["default"],
            feature_pipelines=[],
            detectors=["trend"],
            labelers=[],
            validators=[],
            metrics=["BacktestMetrics"],
            capital=10000.0,
            fee=0.001,
        )
        json_dict = result.to_json_dict()
        assert "config" in json_dict
        assert json_dict["config"]["strategy_id"] == "test"

    def test_to_json_dict_with_window_results(self):
        """to_json_dict() includes window results."""
        result = FlowResult()
        result.window_results = [FlowResult(), FlowResult()]
        json_dict = result.to_json_dict()
        assert "window_results" in json_dict
        assert len(json_dict["window_results"]) == 2

    def test_save_artifacts(self, tmp_path):
        """save_artifacts() creates files."""
        result = FlowResult()
        result.features = pl.DataFrame({"a": [1, 2, 3]})
        result.signals = Signals(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1)],
                    "pair": ["BTCUSDT"],
                    "signal": [1],
                }
            )
        )
        result.labels = pl.DataFrame({"label": [1, 0, 1]})
        result.predictions = pl.DataFrame({"pred": [1, 0, 1]})
        result.trades = pl.DataFrame({"pnl": [100.0, -50.0]})
        result.flow_config = FlowConfig(
            strategy_id="test",
            run_mode=RunMode.SINGLE,
            data_sources=[],
            feature_pipelines=[],
            detectors=[],
            labelers=[],
            validators=[],
            metrics=[],
            capital=10000.0,
            fee=0.001,
        )

        result.save_artifacts(tmp_path)

        assert (tmp_path / "features.parquet").exists()
        assert (tmp_path / "signals.parquet").exists()
        assert (tmp_path / "labels.parquet").exists()
        assert (tmp_path / "predictions.parquet").exists()
        assert (tmp_path / "trades.parquet").exists()
        assert (tmp_path / "flow_config.json").exists()


# ===========================================================================
# FlowConfig Tests
# ===========================================================================


class TestFlowConfig:
    """Tests for FlowConfig dataclass."""

    def test_create_config(self):
        """FlowConfig stores all fields."""
        config = FlowConfig(
            strategy_id="my_flow",
            run_mode=RunMode.WALK_FORWARD,
            data_sources=["binance", "okx"],
            feature_pipelines=["momentum"],
            detectors=["trend", "volume"],
            labelers=["triple_barrier"],
            validators=["lgbm"],
            metrics=["BacktestMetrics"],
            capital=50000.0,
            fee=0.0005,
        )
        assert config.strategy_id == "my_flow"
        assert config.run_mode == RunMode.WALK_FORWARD
        assert len(config.data_sources) == 2
        assert config.capital == 50000.0


# ===========================================================================
# RunMode Tests
# ===========================================================================


class TestRunMode:
    """Tests for RunMode enum."""

    def test_values(self):
        """RunMode has expected values."""
        assert RunMode.SINGLE.value == "single"
        assert RunMode.TEMPORAL_CV.value == "temporal_cv"
        assert RunMode.WALK_FORWARD.value == "walk_forward"
        assert RunMode.LIVE.value == "live"


# ===========================================================================
# AggregationMode Tests
# ===========================================================================


class TestAggregationMode:
    """Tests for AggregationMode enum."""

    def test_values(self):
        """AggregationMode has expected values."""
        assert AggregationMode.MERGE.value == "merge"
        assert AggregationMode.MAJORITY.value == "majority"
        assert AggregationMode.WEIGHTED.value == "weighted"
        assert AggregationMode.UNANIMOUS.value == "unanimous"
        assert AggregationMode.ANY.value == "any"
        assert AggregationMode.META_LABELING.value == "meta_labeling"


# ===========================================================================
# FlowBuilder Factory Tests
# ===========================================================================


class TestFlowFactory:
    """Tests for flow() factory function."""

    def test_returns_builder(self):
        """flow() returns FlowBuilder instance."""
        builder = flow()
        assert isinstance(builder, FlowBuilder)

    def test_default_strategy_id(self):
        """Default strategy_id is 'flow'."""
        builder = flow()
        assert builder.strategy_id == "flow"

    def test_custom_strategy_id(self):
        """Custom strategy_id is set."""
        builder = flow("my_flow")
        assert builder.strategy_id == "my_flow"


# ===========================================================================
# FlowBuilder Fluent API Tests
# ===========================================================================


class TestFlowBuilderFluent:
    """Tests for fluent API methods."""

    def test_data_returns_self(self, sample_raw_data):
        """data() returns self for chaining."""
        builder = flow()
        result = builder.data(raw=sample_raw_data)
        assert result is builder

    def test_features_returns_self(self, sample_raw_data):
        """features() returns self for chaining."""
        builder = flow().data(raw=sample_raw_data)
        with patch.object(builder, "_feature_pipelines", {}):
            mock_pipeline = MagicMock()
            result = builder.features(mock_pipeline)
            assert result is builder

    def test_detector_returns_self(self, sample_raw_data):
        """detector() returns self for chaining."""
        builder = flow().data(raw=sample_raw_data)
        result = builder.detector("example/sma_cross")
        assert result is builder

    def test_signals_returns_self(self, sample_signals):
        """signals() returns self for chaining."""
        builder = flow()
        result = builder.signals(sample_signals)
        assert result is builder
        assert builder._signals is sample_signals

    def test_aggregate_returns_self(self):
        """aggregate() returns self for chaining."""
        builder = flow()
        result = builder.aggregate(mode=AggregationMode.MAJORITY)
        assert result is builder

    def test_labeler_returns_self(self):
        """labeler() returns self for chaining."""
        builder = flow()
        mock_labeler = MagicMock()
        result = builder.labeler(mock_labeler)
        assert result is builder

    def test_validator_returns_self(self):
        """validator() returns self for chaining."""
        builder = flow()
        mock_validator = MagicMock()
        result = builder.validator(mock_validator)
        assert result is builder

    def test_entry_returns_self(self):
        """entry() returns self for chaining."""
        builder = flow()
        result = builder.entry(size=100)
        assert result is builder

    def test_exit_returns_self(self):
        """exit() returns self for chaining."""
        builder = flow()
        result = builder.exit(tp=0.02, sl=0.01)
        assert result is builder

    def test_capital_returns_self(self):
        """capital() returns self for chaining."""
        builder = flow()
        result = builder.capital(50000)
        assert result is builder
        assert builder._capital == 50000

    def test_fee_returns_self(self):
        """fee() returns self for chaining."""
        builder = flow()
        result = builder.fee(0.002)
        assert result is builder
        assert builder._fee == 0.002

    def test_metrics_returns_self(self):
        """metrics() returns self for chaining."""
        builder = flow()
        result = builder.metrics(BacktestMetrics())
        assert result is builder
        assert len(builder._metric_nodes) == 1

    def test_artifacts_returns_self(self, tmp_path):
        """artifacts() returns self for chaining."""
        builder = flow()
        result = builder.artifacts(tmp_path)
        assert result is builder
        assert builder._artifacts_dir == tmp_path


# ===========================================================================
# FlowBuilder Data Configuration Tests
# ===========================================================================


class TestFlowBuilderData:
    """Tests for data configuration."""

    def test_data_with_raw(self, sample_raw_data):
        """data() accepts RawData instance."""
        builder = flow().data(raw=sample_raw_data)
        assert builder._raw is sample_raw_data

    def test_data_with_name(self, sample_raw_data):
        """data(name=...) stores in _named_data."""
        builder = flow().data(raw=sample_raw_data, name="1m")
        assert "1m" in builder._named_data
        assert builder._named_data["1m"] is sample_raw_data

    def test_data_with_params_default_name(self):
        """data() without name uses 'default' key."""
        builder = flow().data(
            store="binance.duckdb",
            pairs=["BTCUSDT"],
            start="2024-01-01",
        )
        assert "default" in builder._named_data

    def test_data_with_params_named(self):
        """data(name=...) stores params dict."""
        builder = flow().data(
            name="binance",
            store="binance.duckdb",
            pairs=["BTCUSDT"],
            start="2024-01-01",
        )
        assert "binance" in builder._named_data
        assert isinstance(builder._named_data["binance"], dict)


# ===========================================================================
# FlowBuilder Detector Configuration Tests
# ===========================================================================


class TestFlowBuilderDetector:
    """Tests for detector configuration."""

    def test_detector_from_registry(self, sample_raw_data):
        """detector() accepts registry name."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross")
        assert "default" in builder._named_detectors

    def test_detector_from_instance(self, sample_raw_data):
        """detector() accepts detector instance."""
        from signalflow.detector import ExampleSmaCrossDetector

        detector = ExampleSmaCrossDetector()
        builder = flow().data(raw=sample_raw_data).detector(detector)
        assert "default" in builder._named_detectors
        assert builder._named_detectors["default"] is detector

    def test_detector_with_name(self, sample_raw_data):
        """detector(name=...) uses explicit name."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross", name="trend")
        assert "trend" in builder._named_detectors

    def test_multiple_detectors(self, sample_raw_data):
        """Multiple detectors get auto-named."""
        builder = (
            flow().data(raw=sample_raw_data).detector("example/sma_cross").detector("example/sma_cross", name="volume")
        )
        assert "default" in builder._named_detectors
        assert "volume" in builder._named_detectors


# ===========================================================================
# FlowBuilder Aggregation Configuration Tests
# ===========================================================================


class TestFlowBuilderAggregate:
    """Tests for aggregation configuration."""

    def test_aggregate_stores_config(self):
        """aggregate() stores configuration."""
        builder = flow().aggregate(
            mode=AggregationMode.WEIGHTED,
            min_agreement=0.6,
            weights=[0.7, 0.3],
        )
        assert builder._aggregation_config is not None
        assert builder._aggregation_config["mode"] == "weighted"
        assert builder._aggregation_config["min_agreement"] == 0.6
        assert builder._aggregation_config["weights"] == [0.7, 0.3]

    def test_aggregate_string_mode(self):
        """aggregate() accepts string mode."""
        builder = flow().aggregate(mode="majority")
        assert builder._aggregation_config["mode"] == "majority"


# ===========================================================================
# FlowBuilder Entry/Exit Configuration Tests
# ===========================================================================


class TestFlowBuilderEntryExit:
    """Tests for entry/exit configuration."""

    def test_entry_stores_config(self):
        """entry() stores configuration."""
        builder = flow().entry(
            rule="market",
            size=200,
            size_pct=0.1,
            max_positions=5,
        )
        assert builder._entry_config["rule"] == "market"
        assert builder._entry_config["size"] == 200
        assert builder._entry_config["size_pct"] == 0.1
        assert builder._entry_config["max_positions"] == 5

    def test_exit_stores_config(self):
        """exit() stores configuration."""
        builder = flow().exit(
            rule="tp_sl",
            tp=0.03,
            sl=0.015,
            trailing=0.02,
        )
        assert builder._exit_config["rule"] == "tp_sl"
        assert builder._exit_config["tp"] == 0.03
        assert builder._exit_config["sl"] == 0.015
        assert builder._exit_config["trailing"] == 0.02


# ===========================================================================
# FlowBuilder Labeler Configuration Tests
# ===========================================================================


class TestFlowBuilderLabeler:
    """Tests for labeler configuration."""

    def test_labeler_from_instance(self):
        """labeler() accepts instance."""
        mock_labeler = MagicMock()
        builder = flow().labeler(mock_labeler)
        assert "default" in builder._named_labelers

    def test_labeler_with_name(self):
        """labeler(name=...) uses explicit name."""
        mock_labeler = MagicMock()
        builder = flow().labeler(mock_labeler, name="triple_barrier")
        assert "triple_barrier" in builder._named_labelers


# ===========================================================================
# FlowBuilder Validator Configuration Tests
# ===========================================================================


class TestFlowBuilderValidator:
    """Tests for validator configuration."""

    def test_validator_from_instance(self):
        """validator() accepts instance."""
        mock_validator = MagicMock()
        builder = flow().validator(mock_validator)
        assert "default" in builder._named_validators

    def test_validator_with_name(self):
        """validator(name=...) uses explicit name."""
        mock_validator = MagicMock()
        builder = flow().validator(mock_validator, name="lgbm")
        assert "lgbm" in builder._named_validators


# ===========================================================================
# FlowBuilder Metrics Configuration Tests
# ===========================================================================


class TestFlowBuilderMetrics:
    """Tests for metrics configuration."""

    def test_metrics_adds_nodes(self):
        """metrics() adds metric nodes."""
        builder = flow().metrics(
            FeatureMetrics(),
            SignalMetrics(),
            BacktestMetrics(),
        )
        assert len(builder._metric_nodes) == 3

    def test_multiple_metrics_calls(self):
        """Multiple metrics() calls accumulate."""
        builder = flow().metrics(FeatureMetrics()).metrics(BacktestMetrics())
        assert len(builder._metric_nodes) == 2


# ===========================================================================
# FlowBuilder Parse Period Tests
# ===========================================================================


class TestFlowBuilderParsePeriod:
    """Tests for _parse_period helper."""

    def test_parse_days_int(self):
        """Parses integer as days."""
        td = FlowBuilder._parse_period(30)
        assert td == timedelta(days=30)

    def test_parse_days_string(self):
        """Parses '30d' as 30 days."""
        td = FlowBuilder._parse_period("30d")
        assert td == timedelta(days=30)

    def test_parse_months(self):
        """Parses '6M' as ~180 days."""
        td = FlowBuilder._parse_period("6M")
        assert td == timedelta(days=180)

    def test_parse_years(self):
        """Parses '1Y' as ~365 days."""
        td = FlowBuilder._parse_period("1Y")
        assert td == timedelta(days=365)

    def test_parse_plain_number(self):
        """Parses plain number string as days."""
        td = FlowBuilder._parse_period("30")
        assert td == timedelta(days=30)

    def test_parse_none_raises(self):
        """None raises ConfigurationError."""
        from signalflow.api.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            FlowBuilder._parse_period(None)


# ===========================================================================
# FlowBuilder Filter Raw Data Tests
# ===========================================================================


class TestFlowBuilderFilterRawData:
    """Tests for _filter_raw_data helper."""

    def test_filters_by_timestamp(self, sample_raw_data):
        """Filters DataFrame by timestamp range."""
        start = datetime(2024, 1, 1, 10)
        end = datetime(2024, 1, 1, 20)

        filtered = FlowBuilder._filter_raw_data(sample_raw_data, start, end)

        df = filtered.get("spot")
        assert df.height > 0
        assert df.select("timestamp").min().item() >= start
        assert df.select("timestamp").max().item() < end


# ===========================================================================
# FlowBuilder Run Tests
# ===========================================================================


class TestFlowBuilderRun:
    """Tests for run() method."""

    def test_run_single_mode(self, sample_raw_data):
        """run() in single mode returns FlowResult."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross")
        # Mock _run_backtest to avoid issues with BacktestResult
        with patch.object(builder, "_run_backtest", return_value=None):
            result = builder.run(mode=RunMode.SINGLE)
        assert isinstance(result, FlowResult)
        assert result.flow_config is not None
        assert result.flow_config.run_mode == RunMode.SINGLE

    def test_run_string_mode(self, sample_raw_data):
        """run() accepts string mode."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross")
        with patch.object(builder, "_run_backtest", return_value=None):
            result = builder.run(mode="single")
        assert result.flow_config.run_mode == RunMode.SINGLE

    def test_run_saves_artifacts(self, sample_raw_data, tmp_path):
        """run() saves artifacts if configured."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross").artifacts(tmp_path)
        with patch.object(builder, "_run_backtest", return_value=None):
            builder.run()
        # At minimum, flow_config.json should be created
        assert (tmp_path / "flow_config.json").exists()

    def test_run_records_execution_time(self, sample_raw_data):
        """run() records execution time."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross")
        with patch.object(builder, "_run_backtest", return_value=None):
            result = builder.run()
        assert result.execution_time > 0

    def test_run_with_metrics_nodes(self, sample_raw_data):
        """run() populates metric results based on configured nodes."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross").metrics(SignalMetrics())
        with patch.object(builder, "_run_backtest", return_value=None):
            result = builder.run()
        assert result.signal_metrics is not None
        assert "total_rows" in result.signal_metrics


# ===========================================================================
# FlowBuilder Repr Tests
# ===========================================================================


class TestFlowBuilderRepr:
    """Tests for __repr__."""

    def test_repr_basic(self):
        """__repr__ shows strategy_id."""
        builder = flow("my_flow")
        repr_str = repr(builder)
        assert "FlowBuilder" in repr_str
        assert "my_flow" in repr_str

    def test_repr_with_components(self, sample_raw_data):
        """__repr__ shows configured components."""
        builder = (
            flow("test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .metrics(BacktestMetrics())
        )
        repr_str = repr(builder)
        assert "trend" in repr_str
        assert "BacktestMetrics" in repr_str


# ===========================================================================
# Metric Node Tests
# ===========================================================================


class TestMetricNodes:
    """Tests for metric node dataclasses."""

    def test_feature_metrics(self):
        """FeatureMetrics has correct defaults."""
        metrics = FeatureMetrics()
        assert metrics.include_correlation is True
        assert metrics.include_importance is True

    def test_signal_metrics(self):
        """SignalMetrics has correct defaults."""
        metrics = SignalMetrics()
        assert metrics.include_frequency is True
        assert metrics.include_clustering is True

    def test_label_metrics(self):
        """LabelMetrics has correct defaults."""
        metrics = LabelMetrics()
        assert metrics.include_distribution is True
        assert metrics.include_holding_time is True

    def test_validation_metrics(self):
        """ValidationMetrics has correct defaults."""
        metrics = ValidationMetrics()
        assert metrics.include_confusion_matrix is True
        assert metrics.include_feature_importance is True

    def test_backtest_metrics(self):
        """BacktestMetrics has correct defaults."""
        metrics = BacktestMetrics()
        assert metrics.include_equity_curve is True
        assert metrics.include_drawdown is True

    def test_live_metrics(self):
        """LiveMetrics has correct defaults."""
        metrics = LiveMetrics()
        assert metrics.include_latency is True
        assert metrics.include_slippage is True


# ===========================================================================
# FlowBuilder Internal Helpers Tests
# ===========================================================================


class TestFlowBuilderHelpers:
    """Tests for internal helper methods."""

    def test_resolve_data_from_raw(self, sample_raw_data):
        """_resolve_data returns _raw if set."""
        builder = flow().data(raw=sample_raw_data)
        resolved = builder._resolve_data()
        assert resolved is sample_raw_data

    def test_resolve_data_missing_raises(self):
        """_resolve_data raises MissingDataError if no data."""
        from signalflow.api.exceptions import MissingDataError

        builder = flow()
        with pytest.raises(MissingDataError):
            builder._resolve_data()

    def test_resolve_signals_empty(self, sample_raw_data):
        """_resolve_signals returns empty Signals if no detectors."""
        builder = flow().data(raw=sample_raw_data)
        signals = builder._resolve_signals(sample_raw_data)
        assert isinstance(signals, Signals)
        assert signals.value.height == 0

    def test_resolve_signals_precomputed(self, sample_raw_data, sample_signals):
        """_resolve_signals returns pre-computed signals."""
        builder = flow().data(raw=sample_raw_data).signals(sample_signals)
        signals = builder._resolve_signals(sample_raw_data)
        assert signals is sample_signals

    def test_compute_feature_metrics(self, sample_raw_data):
        """_compute_feature_metrics returns dict."""
        builder = flow()
        df = sample_raw_data.get("spot")
        result = builder._compute_feature_metrics(df, FeatureMetrics())
        assert "n_features" in result
        assert "n_rows" in result
        assert "null_pct" in result

    def test_compute_signal_metrics(self, sample_signals):
        """_compute_signal_metrics returns dict."""
        builder = flow()
        result = builder._compute_signal_metrics(sample_signals, SignalMetrics())
        assert "total_rows" in result
        assert "active_signals" in result
        assert "signal_rate" in result

    def test_compute_label_metrics(self):
        """_compute_label_metrics returns dict."""
        builder = flow()
        labels = pl.DataFrame({"label": [1, 0, 1, 1, 0]})
        result = builder._compute_label_metrics(labels, LabelMetrics())
        assert "total_labels" in result
        assert "wins" in result
        assert "win_rate" in result

    def test_compute_validation_metrics(self):
        """_compute_validation_metrics returns dict."""
        builder = flow()
        predictions = pl.DataFrame({"pred": [1, 0, 1]})
        result = builder._compute_validation_metrics(predictions, ValidationMetrics())
        assert "n_predictions" in result


# ===========================================================================
# FlowBuilder Aggregate Results Tests
# ===========================================================================


class TestFlowBuilderAggregateResults:
    """Tests for _aggregate_results method."""

    def test_aggregate_empty(self):
        """_aggregate_results handles empty list."""
        result = FlowBuilder._aggregate_results([])
        assert result == {}

    def test_aggregate_single(self):
        """_aggregate_results handles single result."""
        sub = FlowResult()
        sub.backtest_metrics = {"sharpe_ratio": 1.5, "total_return": 0.1, "win_rate": 0.6, "n_trades": 10}
        result = FlowBuilder._aggregate_results([sub])
        assert result["n_folds"] == 1
        assert result["sharpe_avg"] == 1.5
        assert result["n_trades"] == 10

    def test_aggregate_multiple(self):
        """_aggregate_results averages across folds."""
        sub1 = FlowResult()
        sub1.backtest_metrics = {"sharpe_ratio": 1.0, "total_return": 0.1, "win_rate": 0.5, "n_trades": 10}
        sub2 = FlowResult()
        sub2.backtest_metrics = {"sharpe_ratio": 2.0, "total_return": 0.2, "win_rate": 0.7, "n_trades": 20}
        result = FlowBuilder._aggregate_results([sub1, sub2])
        assert result["n_folds"] == 2
        assert result["sharpe_avg"] == 1.5
        assert result["n_trades"] == 30


# ===========================================================================
# Additional FlowResult Tests
# ===========================================================================


class TestFlowResultAdditional:
    """Additional tests for FlowResult."""

    def test_to_json_dict_with_fold_results(self):
        """to_json_dict() includes fold results."""
        result = FlowResult()
        result.fold_results = [FlowResult(), FlowResult()]
        json_dict = result.to_json_dict()
        assert "fold_results" in json_dict
        assert len(json_dict["fold_results"]) == 2

    def test_save_artifacts_with_window_results(self, tmp_path):
        """save_artifacts() saves window results JSON."""
        result = FlowResult()
        result.flow_config = FlowConfig(
            strategy_id="test",
            run_mode=RunMode.WALK_FORWARD,
            data_sources=[],
            feature_pipelines=[],
            detectors=[],
            labelers=[],
            validators=[],
            metrics=[],
            capital=10000.0,
            fee=0.001,
        )

        # Add window results
        wr1 = FlowResult()
        wr1.backtest_metrics = {"sharpe": 1.5}
        wr2 = FlowResult()
        wr2.backtest_metrics = {"sharpe": 2.0}
        result.window_results = [wr1, wr2]

        result.save_artifacts(tmp_path)

        assert (tmp_path / "window_results.json").exists()
        assert (tmp_path / "flow_config.json").exists()

    def test_save_artifacts_with_fold_results(self, tmp_path):
        """save_artifacts() saves fold results JSON."""
        result = FlowResult()
        result.flow_config = FlowConfig(
            strategy_id="test",
            run_mode=RunMode.TEMPORAL_CV,
            data_sources=[],
            feature_pipelines=[],
            detectors=[],
            labelers=[],
            validators=[],
            metrics=[],
            capital=10000.0,
            fee=0.001,
        )

        # Add fold results
        fr1 = FlowResult()
        fr1.backtest_metrics = {"sharpe": 1.0}
        fr2 = FlowResult()
        fr2.backtest_metrics = {"sharpe": 1.5}
        result.fold_results = [fr1, fr2]

        result.save_artifacts(tmp_path)

        assert (tmp_path / "fold_results.json").exists()

    def test_format_metrics_non_dict_non_object(self):
        """_format_metrics handles simple value."""
        result = FlowResult()
        formatted = result._format_metrics("simple string")
        assert formatted == {"value": "simple string"}


# ===========================================================================
# Additional FlowBuilder Tests
# ===========================================================================


class TestFlowBuilderAdditional:
    """Additional tests for FlowBuilder."""

    def test_filter_raw_data_with_nested_structure(self, sample_raw_data):
        """_filter_raw_data handles nested multi-source structure."""
        from signalflow.core import RawData

        # Create RawData with nested structure
        base = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 100,
                "timestamp": [base + timedelta(hours=i) for i in range(100)],
                "open": [50000.0 + i for i in range(100)],
                "high": [50100.0 + i for i in range(100)],
                "low": [49900.0 + i for i in range(100)],
                "close": [50050.0 + i for i in range(100)],
                "volume": [1000.0] * 100,
            }
        )

        nested_raw = RawData(
            datetime_start=base,
            datetime_end=base + timedelta(hours=100),
            pairs=["BTCUSDT"],
            data={
                "spot": {
                    "binance": df,
                    "okx": df,
                }
            },
        )

        start = base + timedelta(hours=10)
        end = base + timedelta(hours=20)

        filtered = FlowBuilder._filter_raw_data(nested_raw, start, end)
        assert "spot" in filtered.data
        assert isinstance(filtered.data["spot"], dict)

    def test_resolve_data_from_named_raw_data(self, sample_raw_data):
        """_resolve_data handles named RawData in _named_data."""
        builder = flow().data(raw=sample_raw_data, name="binance")
        resolved = builder._resolve_data()
        assert resolved is sample_raw_data

    def test_features_with_pipeline_instance(self, sample_raw_data):
        """features() accepts pipeline instance."""
        mock_pipeline = MagicMock()
        builder = flow().data(raw=sample_raw_data).features(mock_pipeline)
        assert "default" in builder._feature_pipelines
        assert builder._feature_pipelines["default"] is mock_pipeline

    def test_features_with_name(self, sample_raw_data):
        """features() accepts name parameter."""
        mock_pipeline = MagicMock()
        builder = flow().data(raw=sample_raw_data).features(mock_pipeline, name="momentum")
        assert "momentum" in builder._feature_pipelines

    def test_compute_labels_single_labeler(self, sample_raw_data, sample_signals):
        """_compute_labels works with single labeler."""
        builder = flow()
        mock_labeler = MagicMock()
        mock_labeler.compute.return_value = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "label": [1],
            }
        )
        builder._named_labelers = {"default": mock_labeler}

        labels = builder._compute_labels(sample_raw_data, sample_signals)
        assert labels is not None
        assert "label" in labels.columns

    def test_repr_with_all_components(self, sample_raw_data):
        """__repr__ shows all component types."""
        mock_labeler = MagicMock()
        mock_validator = MagicMock()

        builder = (
            flow("full_test")
            .data(raw=sample_raw_data)
            .detector("example/sma_cross", name="trend")
            .labeler(mock_labeler, name="triple")
            .validator(mock_validator, name="lgbm")
            .metrics(FeatureMetrics(), SignalMetrics())
        )

        repr_str = repr(builder)
        assert "full_test" in repr_str
        assert "trend" in repr_str
        assert "triple" in repr_str
        assert "lgbm" in repr_str

    def test_resolve_signals_single_detector(self, sample_raw_data):
        """_resolve_signals with single detector."""
        builder = flow().data(raw=sample_raw_data).detector("example/sma_cross")
        signals = builder._resolve_signals(sample_raw_data)
        assert isinstance(signals, Signals)

    def test_compute_features_single_pipeline(self, sample_raw_data):
        """_compute_features transforms data through pipeline."""
        builder = flow()
        mock_pipeline = MagicMock()
        mock_pipeline.compute.return_value = sample_raw_data.get("spot")
        builder._feature_pipelines = {"default": mock_pipeline}

        result = builder._compute_features(sample_raw_data)
        mock_pipeline.compute.assert_called_once()
        assert result is not None
