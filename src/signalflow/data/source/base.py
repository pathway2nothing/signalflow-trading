"""Source plugin contract."""


from typing import Protocol, runtime_checkable

import polars as pl

CANONICAL_COLUMNS = ["pair", "ts", "open", "high", "low", "close", "volume"]
"""Columns every source must return (spot-first; extra columns are allowed)."""


@runtime_checkable
class Source(Protocol):
    """Fetches market data as canonical OHLCV rows."""

    name: str

    def fetch(
        self,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1m",
    ) -> pl.DataFrame:
        """Return a frame with at least :data:`CANONICAL_COLUMNS`, sorted by (pair, ts)."""
        ...


def validate_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Check canonical columns are present and the frame is sorted by (pair, ts)."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"source frame missing columns: {missing}")
    return df.sort(["pair", "ts"])
