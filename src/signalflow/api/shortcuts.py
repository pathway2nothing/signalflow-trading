"""
Quick shortcuts for common SignalFlow operations.

These are thin wrappers for simple use cases. For more control,
use the Builder API or underlying classes directly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from signalflow.core import RawData

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.api.result import BacktestResult


def load(
    source: str | Path,
    *,
    pairs: list[str],
    start: str | datetime,
    end: str | datetime | None = None,
    timeframe: str = "1h",
    data_type: str = "spot",
) -> RawData:
    """
    Load market data from a DuckDB file or exchange cache.

    This is a convenience function for quickly loading data.
    For more control, use RawDataFactory or store classes directly.

    Args:
        source: Path to DuckDB file (string ending in .duckdb or Path object)
        pairs: List of trading pairs to load
        start: Start date (ISO string like "2024-01-01" or datetime)
        end: End date (default: now)
        timeframe: Candle timeframe (default: "1h") - used for exchange cache
        data_type: Data type ("spot", "futures", "perpetual")

    Returns:
        RawData container with loaded data

    Raises:
        ValueError: If source format is invalid or data not found
        FileNotFoundError: If DuckDB file doesn't exist

    Example:
        >>> import signalflow as sf
        >>>
        >>> # Load from DuckDB file
        >>> raw = sf.load(
        ...     "data/binance.duckdb",
        ...     pairs=["BTCUSDT", "ETHUSDT"],
        ...     start="2024-01-01",
        ...     end="2024-06-01",
        ... )
        >>>
        >>> # Access data
        >>> df = raw.spot.to_polars()
        >>> print(f"Loaded {df.height} bars")
    """
    from signalflow.data import RawDataFactory

    # Parse dates
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end) if end else datetime.now()

    if start_dt is None:
        raise ValueError("start date is required")

    # Determine source type
    source_path = Path(source) if isinstance(source, str) else source

    # DuckDB file path
    if str(source_path).endswith(".duckdb"):
        if not source_path.exists():
            raise FileNotFoundError(f"DuckDB file not found: {source_path}")

        # Use appropriate factory method based on data_type
        if data_type == "spot":
            return RawDataFactory.from_duckdb_spot_store(
                spot_store_path=source_path,
                pairs=pairs,
                start=start_dt,
                end=end_dt,
            )
        else:
            # For non-spot data, use generic store approach
            from signalflow.data import StoreFactory

            store = StoreFactory.create_raw_store(
                backend="duckdb",
                data_type=data_type,
                db_path=source_path,
            )
            return RawDataFactory.from_stores(
                stores=[store],
                pairs=pairs,
                start=start_dt,
                end=end_dt,
            )

    # Exchange name - not directly supported yet
    # User should download data first using data loaders
    raise ValueError(
        f"Direct exchange loading not yet supported. "
        f"Please download data first using data loaders:\n"
        f"  from signalflow.data.source import BinanceDataSource\n"
        f"  source = BinanceDataSource(...)\n"
        f"  source.download(pairs={pairs}, start=..., end=...)\n"
        f"Then load from the DuckDB file."
    )


def backtest(
    detector: SignalDetector | str,
    *,
    raw: RawData | None = None,
    pairs: list[str] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    tp: float = 0.02,
    sl: float = 0.01,
    capital: float = 10_000.0,
    **kwargs,
) -> BacktestResult:
    """
    Quick backtest with minimal configuration.

    This is a convenience function for simple backtests.
    For more control, use sf.Backtest() builder.

    Args:
        detector: SignalDetector instance or registry name (e.g., "example/sma_cross")
        raw: Pre-loaded RawData (alternative to pairs/start/end)
        pairs: Trading pairs (if raw not provided)
        start: Start date (if raw not provided)
        end: End date (if raw not provided)
        tp: Take profit percentage (default: 2%)
        sl: Stop loss percentage (default: 1%)
        capital: Initial capital (default: 10,000)
        **kwargs: Additional parameters for detector (if using registry name)

    Returns:
        BacktestResult with trades, metrics, and analytics

    Raises:
        ValueError: If neither raw nor pairs/start provided

    Example:
        >>> import signalflow as sf
        >>> from signalflow.detector import ExampleSmaCrossDetector
        >>>
        >>> # With pre-loaded data
        >>> result = sf.backtest(
        ...     detector=ExampleSmaCrossDetector(fast_period=20, slow_period=50),
        ...     raw=my_raw_data,
        ...     tp=0.03,
        ...     sl=0.015,
        ... )
        >>>
        >>> # With registry name
        >>> result = sf.backtest(
        ...     detector="example/sma_cross",
        ...     raw=my_raw_data,
        ...     fast_period=20,
        ...     slow_period=50,
        ... )
        >>>
        >>> print(result.summary())
    """
    from signalflow.api.builder import Backtest

    builder = Backtest()

    # Configure data
    if raw is not None:
        builder.data(raw=raw)
    elif pairs and start:
        builder.data(pairs=pairs, start=start, end=end)
    else:
        raise ValueError("Either 'raw' or 'pairs' + 'start' must be provided")

    # Configure detector
    if isinstance(detector, str):
        builder.detector(detector, **kwargs)
    else:
        builder.detector(detector)

    # Configure exit rules
    builder.exit(tp=tp, sl=sl)

    # Configure capital
    builder.capital(capital)

    return builder.run()


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse string to datetime, or return datetime as-is."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # Try ISO format parsing
    return datetime.fromisoformat(value)
