from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from signalflow.core import SfComponentType


@dataclass
class RawDataSource(ABC):
    """Abstract base class for raw data sources.

    Defines the interface for data sources that provide market data
    (exchanges, APIs, files, etc.). Sources are passive - they define
    where data comes from but don't handle downloading.

    RawDataSource is typically used in combination with RawDataLoader,
    which handles the actual data retrieval and storage logic.

    Key responsibilities:
        - Define connection parameters (API keys, endpoints, etc.)
        - Provide authentication credentials
        - Specify data source configuration

    Common implementations:
        - Exchange APIs (Binance, Coinbase, etc.)
        - Data providers (CryptoCompare, CoinGecko, etc.)
        - File sources (CSV, Parquet, etc.)
        - Database connections

    Attributes:
        component_type (ClassVar[SfComponentType]): Always RAW_DATA_SOURCE for registry.

    Example:
        ```python
        from signalflow.core import sf_component, SfComponentType
        from dataclasses import dataclass

        @dataclass
        @sf_component(name="binance_spot")
        class BinanceSpotSource(RawDataSource):
            '''Binance Spot API source'''
            api_key: str = ""
            api_secret: str = ""
            base_url: str = "https://api.binance.com"

            def get_client(self):
                '''Create authenticated client'''
                from binance.client import Client
                return Client(self.api_key, self.api_secret)

        # Use with loader
        source = BinanceSpotSource(
            api_key="your_key",
            api_secret="your_secret"
        )

        loader = BinanceSpotLoader(
            source=source,
            store=store
        )
        ```

    Example:
        ```python
        # File-based source
        @dataclass
        @sf_component(name="csv_source")
        class CsvSource(RawDataSource):
            '''CSV file source'''
            file_path: Path
            separator: str = ","

            def read(self) -> pl.DataFrame:
                return pl.read_csv(self.file_path, separator=self.separator)

        # Database source
        @dataclass
        @sf_component(name="postgres_source")
        class PostgresSource(RawDataSource):
            '''PostgreSQL database source'''
            host: str
            port: int = 5432
            database: str
            user: str
            password: str

            def get_connection_string(self) -> str:
                return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        ```

    Note:
        Source classes are typically passive configuration containers.
        Active data retrieval is handled by RawDataLoader implementations.
        Use @sf_component decorator to register sources in the registry.

    See Also:
        RawDataLoader: Active component that uses sources to download data.
        RawDataStore: Storage backend for persisting downloaded data.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.RAW_DATA_SOURCE


class RawDataLoader(ABC):
    """Abstract base class for raw data loaders.

    Defines the interface for loading market data from sources and
    storing it in persistent storage. Loaders orchestrate the data
    pipeline: source → transformation → storage.

    Key responsibilities:
        - Download data from sources (download)
        - Sync/update existing data (sync)
        - Handle rate limiting and retries
        - Transform data to canonical format
        - Store data in persistent backend

    Typical workflow:
        1. Initial download: Fetch historical data
        2. Incremental sync: Update with latest data
        3. Gap filling: Detect and fill missing periods
        4. Validation: Ensure data quality and completeness

    Attributes:
        component_type (ClassVar[SfComponentType]): Always RAW_DATA_LOADER for registry.

    Example:
        ```python
        from signalflow.core import sf_component, SfComponentType
        from signalflow.data.raw_store import DuckDbSpotStore
        from datetime import datetime

        @sf_component(name="binance_spot_loader")
        class BinanceSpotLoader(RawDataLoader):
            '''Loads Binance spot data'''

            def __init__(self, source: BinanceSpotSource, store: DuckDbSpotStore):
                self.source = source
                self.store = store
                self.client = source.get_client()

            def download(self, pairs: list[str], start: datetime, end: datetime):
                '''Download historical data'''
                for pair in pairs:
                    klines = self.client.get_historical_klines(
                        symbol=pair,
                        interval="1m",
                        start_str=start.isoformat(),
                        end_str=end.isoformat()
                    )

                    # Transform to canonical format
                    formatted = self._format_klines(klines)

                    # Store
                    self.store.insert_klines(pair, formatted)

            def sync(self, pairs: list[str]):
                '''Sync latest data'''
                for pair in pairs:
                    # Get last timestamp
                    _, max_ts = self.store.get_time_bounds(pair)

                    if max_ts:
                        # Fetch data from max_ts to now
                        self.download(
                            pairs=[pair],
                            start=max_ts,
                            end=datetime.now()
                        )

            def _format_klines(self, klines):
                '''Transform to canonical format'''
                return [
                    {
                        "timestamp": datetime.fromtimestamp(k[0] / 1000),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "trades": int(k[8])
                    }
                    for k in klines
                ]

        # Usage
        source = BinanceSpotSource(api_key="key", api_secret="secret")
        store = DuckDbSpotStore(Path("data/binance.duckdb"))
        loader = BinanceSpotLoader(source=source, store=store)

        # Initial download
        loader.download(
            pairs=["BTCUSDT", "ETHUSDT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31)
        )

        # Daily sync
        loader.sync(pairs=["BTCUSDT", "ETHUSDT"])
        ```

    Note:
        Implementations should handle rate limiting and API errors gracefully.
        sync() should be idempotent - safe to call multiple times.
        Consider implementing gap detection and backfilling logic.

    See Also:
        RawDataSource: Defines where data comes from.
        RawDataStore: Defines where data is stored.
        RawDataFactory: Creates RawData from stored data.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.RAW_DATA_LOADER

    @abstractmethod
    def download(self, **kwargs):
        """Download historical data from source to storage.

        Initial data acquisition for a date range. Typically used for:
            - First-time setup with historical data
            - Backfilling large time periods
            - Bulk data imports

        Args:
            **kwargs: Implementation-specific parameters. Common parameters:
                - pairs (list[str]): Trading pairs to download
                - start (datetime): Start datetime
                - end (datetime): End datetime
                - timeframe (str): Candlestick timeframe (e.g., "1m", "1h")

        Example:
            ```python
            # Download historical data
            loader.download(
                pairs=["BTCUSDT", "ETHUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 12, 31),
                timeframe="1m"
            )

            # Download with progress tracking
            loader.download(
                pairs=["BTCUSDT"],
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31),
                batch_size=1000,
                show_progress=True
            )
            ```

        Note:
            Should handle API rate limits with retries/delays.
            Consider implementing chunking for large date ranges.
            Validate data quality before storing.
        """
        pass

    @abstractmethod
    def sync(self, **kwargs):
        """Sync/update existing data with latest data.

        Incremental update for keeping data current. Typically used for:
            - Daily/hourly updates
            - Real-time data synchronization
            - Gap filling

        Implementation should:
            - Detect last available timestamp per pair
            - Fetch missing data from last timestamp to now
            - Handle overlapping data (upsert)
            - Be idempotent (safe to run multiple times)

        Args:
            **kwargs: Implementation-specific parameters. Common parameters:
                - pairs (list[str]): Trading pairs to sync
                - force (bool): Force full resync instead of incremental

        Example:
            ```python
            # Incremental sync (fetch latest data)
            loader.sync(pairs=["BTCUSDT", "ETHUSDT"])

            # Force full resync
            loader.sync(pairs=["BTCUSDT"], force=True)

            # Scheduled sync (e.g., cron job)
            import schedule

            def daily_sync():
                loader.sync(pairs=["BTCUSDT", "ETHUSDT"])
                print(f"Synced at {datetime.now()}")

            schedule.every().day.at("00:00").do(daily_sync)
            ```

        Note:
            Should be idempotent - running multiple times is safe.
            Consider checking for gaps and backfilling if needed.
            Log sync status for monitoring.
        """
        pass
