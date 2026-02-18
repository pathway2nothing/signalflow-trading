"""Artifact schema definitions and validation.

Provides type-safe validation for data flowing between Flow nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import polars as pl


class ArtifactType(StrEnum):
    """Standard artifact types in SignalFlow."""

    OHLCV = "ohlcv"
    SIGNALS = "signals"
    FEATURES = "features"
    LABELS = "labels"
    TRADES = "trades"
    METRICS = "metrics"
    VALIDATED_SIGNALS = "validated_signals"
    TRAINING_SIGNALS = "training_signals"


@dataclass(frozen=True)
class ColumnSchema:
    """Schema for a single column.

    Attributes:
        name: Column name
        dtype: Expected Polars dtype or dtype category ("numeric", "temporal", "string")
        nullable: Whether null values are allowed
        constraints: Additional constraints (e.g., {"min": 0, "max": 1})
    """

    name: str
    dtype: pl.DataType | str
    nullable: bool = True
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactSchema:
    """Schema definition for an artifact type.

    Attributes:
        artifact_type: The artifact type this schema describes
        required_columns: Columns that must be present
        optional_columns: Columns that may be present
        allow_extra: Whether extra columns are allowed
    """

    artifact_type: ArtifactType
    required_columns: tuple[ColumnSchema, ...] = ()
    optional_columns: tuple[ColumnSchema, ...] = ()
    allow_extra: bool = True

    def validate(self, df: pl.DataFrame, strict: bool = False) -> list[str]:
        """Validate a DataFrame against this schema.

        Args:
            df: DataFrame to validate
            strict: If True, raise on first error; else collect all errors

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Check required columns exist
        for col_schema in self.required_columns:
            if col_schema.name not in df.columns:
                msg = f"Missing required column: {col_schema.name}"
                if strict:
                    raise ValueError(msg)
                errors.append(msg)
                continue

            # Check dtype
            actual_dtype = df.schema.get(col_schema.name)
            # Null dtype is acceptable for nullable columns (all-null data)
            if actual_dtype == pl.Null and col_schema.nullable:
                pass  # OK
            elif not self._dtype_compatible(actual_dtype, col_schema.dtype):
                msg = f"Column '{col_schema.name}' has dtype {actual_dtype}, expected {col_schema.dtype}"
                if strict:
                    raise ValueError(msg)
                errors.append(msg)

            # Check nullable constraint
            if not col_schema.nullable and df[col_schema.name].null_count() > 0:
                msg = f"Column '{col_schema.name}' contains nulls but nullable=False"
                if strict:
                    raise ValueError(msg)
                errors.append(msg)

        # Check no unknown columns if allow_extra=False
        if not self.allow_extra:
            known = {c.name for c in self.required_columns} | {c.name for c in self.optional_columns}
            extra = set(df.columns) - known
            if extra:
                msg = f"Unexpected columns: {extra}"
                if strict:
                    raise ValueError(msg)
                errors.append(msg)

        return errors

    @staticmethod
    def _dtype_compatible(actual: pl.DataType | None, expected: pl.DataType | str) -> bool:
        """Check if actual dtype is compatible with expected."""
        if actual is None:
            return False

        if isinstance(expected, str):
            # Category matching
            if expected == "numeric":
                return actual in (
                    pl.Float32,
                    pl.Float64,
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt16,
                    pl.UInt32,
                    pl.UInt64,
                )
            if expected == "temporal":
                return isinstance(actual, (pl.Datetime, pl.Date, pl.Time, pl.Duration))
            if expected == "string":
                return actual in (pl.Utf8, pl.String, pl.Categorical)
            return False

        # Exact or compatible match
        if actual == expected:
            return True

        # Allow Float64 where Float32 expected, etc.
        return isinstance(actual, type(expected))


# ============================================================================
# Standard Schemas
# ============================================================================

OHLCV_SCHEMA = ArtifactSchema(
    artifact_type=ArtifactType.OHLCV,
    required_columns=(
        ColumnSchema("pair", "string", nullable=False),
        ColumnSchema("timestamp", "temporal", nullable=False),
        ColumnSchema("open", "numeric", nullable=False),
        ColumnSchema("high", "numeric", nullable=False),
        ColumnSchema("low", "numeric", nullable=False),
        ColumnSchema("close", "numeric", nullable=False),
        ColumnSchema("volume", "numeric", nullable=False),
    ),
    optional_columns=(ColumnSchema("trades", pl.Int64),),
    allow_extra=True,
)

SIGNALS_SCHEMA = ArtifactSchema(
    artifact_type=ArtifactType.SIGNALS,
    required_columns=(
        ColumnSchema("pair", "string", nullable=False),
        ColumnSchema("timestamp", "temporal", nullable=False),
        ColumnSchema("signal_type", "numeric", nullable=True),  # null = no signal
    ),
    optional_columns=(
        ColumnSchema("signal", "numeric"),
        ColumnSchema("probability", "numeric"),
        ColumnSchema("signal_category", "string"),
    ),
    allow_extra=True,
)

FEATURES_SCHEMA = ArtifactSchema(
    artifact_type=ArtifactType.FEATURES,
    required_columns=(
        ColumnSchema("pair", "string", nullable=False),
        ColumnSchema("timestamp", "temporal", nullable=False),
    ),
    optional_columns=(),
    allow_extra=True,  # Feature columns are dynamic
)

LABELS_SCHEMA = ArtifactSchema(
    artifact_type=ArtifactType.LABELS,
    required_columns=(
        ColumnSchema("pair", "string", nullable=False),
        ColumnSchema("timestamp", "temporal", nullable=False),
        ColumnSchema("label", "numeric", nullable=True),
    ),
    optional_columns=(
        ColumnSchema("returns", "numeric"),
        ColumnSchema("duration", "numeric"),
    ),
    allow_extra=True,
)

TRADES_SCHEMA = ArtifactSchema(
    artifact_type=ArtifactType.TRADES,
    required_columns=(
        ColumnSchema("pair", "string", nullable=False),
        ColumnSchema("timestamp", "temporal", nullable=False),
    ),
    optional_columns=(
        ColumnSchema("pnl", "numeric"),
        ColumnSchema("realized_pnl", "numeric"),
        ColumnSchema("entry_price", "numeric"),
        ColumnSchema("exit_price", "numeric"),
    ),
    allow_extra=True,
)

# Registry of schemas by artifact type
SCHEMA_REGISTRY: dict[str, ArtifactSchema] = {
    "ohlcv": OHLCV_SCHEMA,
    "signals": SIGNALS_SCHEMA,
    "validated_signals": SIGNALS_SCHEMA,
    "training_signals": SIGNALS_SCHEMA,
    "features": FEATURES_SCHEMA,
    "labels": LABELS_SCHEMA,
    "trades": TRADES_SCHEMA,
}


def get_schema(artifact_type: str) -> ArtifactSchema | None:
    """Get schema for an artifact type."""
    return SCHEMA_REGISTRY.get(artifact_type)
