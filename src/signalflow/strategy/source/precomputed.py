"""Signal sources for strategy runner."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Iterator

import polars as pl


@dataclass
class PrecomputedSignalSource:
    """Signal source from precomputed DataFrame.
    
    Used for backtesting with pre-detected signals.
    
    Attributes:
        signals_df: DataFrame with all signals
            Required columns: pair, timestamp, signal_type, signal
        pair_col: Name of pair column
        ts_col: Name of timestamp column
        signal_col: Name of signal column (1=long, -1=short, 0=none)
    """
    signals_df: pl.DataFrame
    pair_col: str = 'pair'
    ts_col: str = 'timestamp'
    signal_col: str = 'signal'
    
    _indexed: dict[datetime, pl.DataFrame] | None = None
    _timestamps: list[datetime] | None = None
    
    def __post_init__(self) -> None:
        self._validate()
        self._build_index()
    
    def _validate(self) -> None:
        """Validate signals DataFrame."""
        required = {self.pair_col, self.ts_col, self.signal_col, 'signal_type'}
        missing = required - set(self.signals_df.columns)
        if missing:
            raise ValueError(f"Signals DataFrame missing columns: {sorted(missing)}")
    
    def _build_index(self) -> None:
        """Build timestamp index for fast lookup."""
        # Filter to only actual signals (non-zero)
        active_signals = self.signals_df.filter(
            pl.col(self.signal_col) != 0
        )
        
        # Group by timestamp
        self._indexed = {}
        if active_signals.height > 0:
            for ts in active_signals.get_column(self.ts_col).unique().sort().to_list():
                ts_signals = active_signals.filter(pl.col(self.ts_col) == ts)
                self._indexed[ts] = ts_signals
        
        # Store sorted timestamps
        self._timestamps = sorted(self._indexed.keys())
    
    def get_signals_at(self, ts: datetime) -> pl.DataFrame:
        """Get signals for specific timestamp."""
        if self._indexed is None:
            self._build_index()
        return self._indexed.get(ts, pl.DataFrame())
    
    def timestamps(self) -> Iterator[datetime]:
        """Iterate over all timestamps with active signals."""
        if self._timestamps is None:
            self._build_index()
        return iter(self._timestamps)
    
    def __len__(self) -> int:
        """Number of timestamps with signals."""
        if self._timestamps is None:
            self._build_index()
        return len(self._timestamps)
