"""Tests for artifact schema validation."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.config.artifact_schema import (
    FEATURES_SCHEMA,
    LABELS_SCHEMA,
    OHLCV_SCHEMA,
    SIGNALS_SCHEMA,
    TRADES_SCHEMA,
    ArtifactSchema,
    ArtifactType,
    ColumnSchema,
    get_schema,
)


class TestColumnSchema:
    """Tests for ColumnSchema."""

    def test_basic_creation(self):
        col = ColumnSchema("price", pl.Float64, nullable=False)
        assert col.name == "price"
        assert col.dtype == pl.Float64
        assert col.nullable is False

    def test_default_nullable(self):
        col = ColumnSchema("value", "numeric")
        assert col.nullable is True

    def test_with_constraints(self):
        col = ColumnSchema("probability", "numeric", constraints={"min": 0, "max": 1})
        assert col.constraints == {"min": 0, "max": 1}


class TestArtifactSchema:
    """Tests for ArtifactSchema validation."""

    def test_valid_ohlcv(self):
        """Valid OHLCV data passes validation."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )
        errors = OHLCV_SCHEMA.validate(df)
        assert errors == []

    def test_missing_column(self):
        """Missing required column produces error."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                # Missing OHLCV columns
            }
        )
        errors = OHLCV_SCHEMA.validate(df)
        assert any("open" in e for e in errors)
        assert any("high" in e for e in errors)
        assert any("low" in e for e in errors)
        assert any("close" in e for e in errors)
        assert any("volume" in e for e in errors)

    def test_wrong_dtype(self):
        """Wrong dtype produces error."""
        df = pl.DataFrame(
            {
                "pair": [123],  # Should be string
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )
        errors = OHLCV_SCHEMA.validate(df)
        assert any("pair" in e and "dtype" in e for e in errors)

    def test_null_in_non_nullable(self):
        """Null in non-nullable column produces error."""
        df = pl.DataFrame(
            {
                "pair": [None],  # Should not be null
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )
        errors = OHLCV_SCHEMA.validate(df)
        assert any("null" in e.lower() for e in errors)

    def test_strict_mode_raises(self):
        """Strict mode raises on first error."""
        df = pl.DataFrame({"invalid": [1]})
        with pytest.raises(ValueError, match="Missing required column"):
            OHLCV_SCHEMA.validate(df, strict=True)

    def test_extra_columns_allowed(self):
        """Extra columns are allowed by default."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
                "extra_col": ["ignored"],
            }
        )
        errors = OHLCV_SCHEMA.validate(df)
        assert errors == []

    def test_extra_columns_forbidden(self):
        """Extra columns can be forbidden."""
        schema = ArtifactSchema(
            artifact_type=ArtifactType.OHLCV,
            required_columns=(ColumnSchema("pair", "string"),),
            allow_extra=False,
        )
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "unexpected": [1],
            }
        )
        errors = schema.validate(df)
        assert any("Unexpected" in e for e in errors)


class TestSignalsSchema:
    """Tests for SIGNALS_SCHEMA."""

    def test_valid_signals(self):
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [1],
            }
        )
        errors = SIGNALS_SCHEMA.validate(df)
        assert errors == []

    def test_null_signal_type_allowed(self):
        """Null signal_type is valid (means no signal)."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [None],
            }
        )
        errors = SIGNALS_SCHEMA.validate(df)
        assert errors == []

    def test_with_probability(self):
        """Signals with probability column."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "signal_type": [1],
                "probability": [0.8],
            }
        )
        errors = SIGNALS_SCHEMA.validate(df)
        assert errors == []


class TestFeaturesSchema:
    """Tests for FEATURES_SCHEMA."""

    def test_valid_features(self):
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "sma_20": [100.0],
                "rsi_14": [50.0],
            }
        )
        errors = FEATURES_SCHEMA.validate(df)
        assert errors == []

    def test_dynamic_columns_allowed(self):
        """Feature columns are dynamic (allow_extra=True)."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "any_feature_name": [1.0],
                "another_feature": [2.0],
            }
        )
        errors = FEATURES_SCHEMA.validate(df)
        assert errors == []


class TestLabelsSchema:
    """Tests for LABELS_SCHEMA."""

    def test_valid_labels(self):
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "label": [1],
            }
        )
        errors = LABELS_SCHEMA.validate(df)
        assert errors == []

    def test_null_label_allowed(self):
        """Null label is valid."""
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "label": [None],
            }
        )
        errors = LABELS_SCHEMA.validate(df)
        assert errors == []


class TestTradesSchema:
    """Tests for TRADES_SCHEMA."""

    def test_valid_trades(self):
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
            }
        )
        errors = TRADES_SCHEMA.validate(df)
        assert errors == []

    def test_with_pnl(self):
        df = pl.DataFrame(
            {
                "pair": ["BTCUSDT"],
                "timestamp": [datetime(2024, 1, 1)],
                "pnl": [100.0],
                "entry_price": [50000.0],
                "exit_price": [51000.0],
            }
        )
        errors = TRADES_SCHEMA.validate(df)
        assert errors == []


class TestSchemaRegistry:
    """Tests for schema registry."""

    def test_get_known_schema(self):
        schema = get_schema("ohlcv")
        assert schema is not None
        assert schema.artifact_type == ArtifactType.OHLCV

    def test_get_signals_schema(self):
        schema = get_schema("signals")
        assert schema is not None
        assert schema.artifact_type == ArtifactType.SIGNALS

    def test_get_validated_signals_schema(self):
        """validated_signals uses same schema as signals."""
        schema = get_schema("validated_signals")
        assert schema is not None
        assert schema.artifact_type == ArtifactType.SIGNALS

    def test_get_unknown_schema(self):
        schema = get_schema("unknown_type")
        assert schema is None


class TestDtypeCompatibility:
    """Tests for dtype compatibility checking."""

    def test_numeric_category(self):
        """Numeric category accepts various numeric types."""
        schema = ArtifactSchema(
            artifact_type=ArtifactType.FEATURES,
            required_columns=(ColumnSchema("value", "numeric"),),
        )

        # Float64
        df = pl.DataFrame({"value": [1.0]})
        assert schema.validate(df) == []

        # Int64
        df = pl.DataFrame({"value": [1]})
        assert schema.validate(df) == []

    def test_temporal_category(self):
        """Temporal category accepts datetime types."""
        schema = ArtifactSchema(
            artifact_type=ArtifactType.FEATURES,
            required_columns=(ColumnSchema("ts", "temporal"),),
        )

        df = pl.DataFrame({"ts": [datetime(2024, 1, 1)]})
        assert schema.validate(df) == []

    def test_string_category(self):
        """String category accepts string types."""
        schema = ArtifactSchema(
            artifact_type=ArtifactType.FEATURES,
            required_columns=(ColumnSchema("name", "string"),),
        )

        df = pl.DataFrame({"name": ["test"]})
        assert schema.validate(df) == []
