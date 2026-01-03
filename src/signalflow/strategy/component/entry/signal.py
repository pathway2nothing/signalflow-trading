from dataclasses import dataclass
from signalflow.core import Signals, Order, StrategyState, SignalType, sf_component
from signalflow.strategy.component.base import EntryRule

import polars as pl

@dataclass
@sf_component(name='signal_entry')
class SignalEntryRule(EntryRule):
    """
    Entry rule based on signals.
    
    Creates long positions for RISE signals and short positions for FALL signals.
    Position size is based on signal probability.
    """
    
    base_position_size: float = 100.0 
    use_probability_sizing: bool = True 
    min_probability: float = 0.5  
    max_positions_per_pair: int = 1  
    allow_shorts: bool = False  
    
    pair_col: str = 'pair'
    ts_col: str = 'timestamp'
    
    def check_entries(
        self,
        signals: Signals,
        prices: dict[str, float],
        state: StrategyState
    ) -> list[Order]:
        orders: list[Order] = []
        
        if signals is None or signals.value.height == 0:
            return orders
        
        positions_by_pair = {}
        for pos in state.portfolio.open_positions():
            positions_by_pair.setdefault(pos.pair, []).append(pos)
        
        df = signals.value
        
        actionable_types = [SignalType.RISE.value]
        if self.allow_shorts:
            actionable_types.append(SignalType.FALL.value)
        
        df = df.filter(pl.col('signal_type').is_in(actionable_types))
        
        if 'probability' in df.columns:
            df = df.filter(pl.col('probability') >= self.min_probability)
        
        for row in df.iter_rows(named=True):
            pair = row[self.pair_col]
            signal_type = row['signal_type']
            probability = row.get('probability', 1.0)
            
            existing_positions = positions_by_pair.get(pair, [])
            if len(existing_positions) >= self.max_positions_per_pair:
                continue
            
            price = prices.get(pair)
            if price is None or price <= 0:
                continue
            
            if signal_type == SignalType.RISE.value:
                side = 'BUY'
            elif signal_type == SignalType.FALL.value and self.allow_shorts:
                side = 'SELL'
            else:
                continue
            
            notional = self.base_position_size
            if self.use_probability_sizing:
                notional *= probability
            
            qty = notional / price
            
            order = Order(
                pair=pair,
                side=side,
                order_type='MARKET',
                qty=qty,
                signal_strength=probability,
                meta={
                    'signal_type': signal_type,
                    'signal_probability': probability,
                    'signal_ts': row.get(self.ts_col),
                }
            )
            orders.append(order)
            
            positions_by_pair.setdefault(pair, []).append(None)
        
        return orders
