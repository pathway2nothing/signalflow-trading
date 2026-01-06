from dataclasses import dataclass, field
from signalflow.core import Signals, Order, StrategyState, SignalType, sf_component
from signalflow.strategy.component.base import EntryRule
import polars as pl

@dataclass
@sf_component(name='fixed_size_entry')
class FixedSizeEntryRule(EntryRule):
    """Simple entry rule with fixed position size."""
    
    position_size: float = 0.01  
    signal_types: list[str] = field(default_factory=lambda: [SignalType.RISE.value])
    max_positions: int = 10  
    
    pair_col: str = 'pair'
    
    def check_entries(
        self,
        signals: Signals,
        prices: dict[str, float],
        state: StrategyState
    ) -> list[Order]:
        orders: list[Order] = []
        
        if signals is None or signals.value.height == 0:
            return orders
        
        open_count = len(state.portfolio.open_positions())
        if open_count >= self.max_positions:
            return orders
        
        df = signals.value.filter(pl.col('signal_type').is_in(self.signal_types))
        
        for row in df.iter_rows(named=True):
            if open_count >= self.max_positions:
                break
                
            pair = row[self.pair_col]
            signal_type = row['signal_type']
            
            price = prices.get(pair)
            if price is None or price <= 0:
                continue
            
            side = 'BUY' if signal_type == SignalType.RISE.value else 'SELL'
            
            order = Order(
                pair=pair,
                side=side,
                order_type='MARKET',
                qty=self.position_size,
                meta={'signal_type': signal_type}
            )
            orders.append(order)
            open_count += 1
        
        return orders