from signalflow.strategy.types import NewPositionOrder, ClosePositionOrder, Position, Trade
from typing import Protocol


class OrderExecutor(Protocol):
    """Protocol for order execution."""
    
    def execute_entry(
        self,
        order: NewPositionOrder,
        fee_rate: float,
    ) -> tuple[Position, Trade]:
        """Execute entry order, return new position and trade."""
        ...
    
    def execute_exit(
        self,
        position: Position,
        order: ClosePositionOrder,
        fee_rate: float,
    ) -> Trade:
        """Execute exit order, return trade."""
        ...


@dataclass 
class OhlcvPriceSource:
    """Price source from OHLCV DataFrame.
    
    Provides prices for any timestamp in the data.
    
    Attributes:
        ohlcv_df: DataFrame with OHLCV data
            Required columns: pair, timestamp, close (and optionally open, high, low)
        price_col: Which price to use ('close', 'open', etc.)
    """
    ohlcv_df: pl.DataFrame
    pair_col: str = 'pair'
    ts_col: str = 'timestamp'
    price_col: str = 'close'
    
    _indexed: dict[datetime, dict[str, float]] | None = None
    
    def __post_init__(self) -> None:
        self._validate()
        self._build_index()
    
    def _validate(self) -> None:
        required = {self.pair_col, self.ts_col, self.price_col}
        missing = required - set(self.ohlcv_df.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {sorted(missing)}")
    
    def _build_index(self) -> None:
        """Build timestamp -> {pair: price} index."""
        self._indexed = {}
        
        for row in self.ohlcv_df.select([
            self.pair_col, self.ts_col, self.price_col
        ]).iter_rows(named=True):
            ts = row[self.ts_col]
            pair = row[self.pair_col]
            price = row[self.price_col]
            
            if ts not in self._indexed:
                self._indexed[ts] = {}
            self._indexed[ts][pair] = float(price)
    
    def get_prices_at(self, ts: datetime) -> dict[str, float]:
        """Get all pair prices at timestamp."""
        if self._indexed is None:
            self._build_index()
        return self._indexed.get(ts, {})
    
    def get_price(self, pair: str, ts: datetime) -> float | None:
        """Get price for specific pair at timestamp."""
        prices = self.get_prices_at(ts)
        return prices.get(pair)