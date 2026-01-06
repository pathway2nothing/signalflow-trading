from __future__ import annotations

from dataclasses import dataclass
from signalflow.core import Signals, Order, StrategyState, SignalType, sf_component, Position
from signalflow.strategy.component.base import EntryRule
import polars as pl


@dataclass
@sf_component(name='signal', override=True)
class SignalEntryRule(EntryRule):

    base_position_size: float = 100.0
    use_probability_sizing: bool = True
    min_probability: float = 0.5
    max_positions_per_pair: int = 1
    max_total_positions: int = 20
    allow_shorts: bool = False
    max_capital_usage: float = 0.95 
    min_order_notional: float = 10.0 
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
        
        positions_by_pair: dict[str, list[Position]] = {}
        for pos in state.portfolio.open_positions():
            positions_by_pair.setdefault(pos.pair, []).append(pos)
        
        total_open = len(state.portfolio.open_positions())
        if total_open >= self.max_total_positions:
            return orders
        
        available_cash = state.portfolio.cash
        
        used_capital = sum(
            pos.entry_price * pos.qty 
            for pos in state.portfolio.open_positions()
        )
        
        total_equity = available_cash + used_capital
        max_allowed_in_positions = total_equity * self.max_capital_usage
        remaining_allocation = max_allowed_in_positions - used_capital
        
        df = signals.value
        actionable_types = [SignalType.RISE.value]
        if self.allow_shorts:
            actionable_types.append(SignalType.FALL.value)
        
        df = df.filter(pl.col('signal_type').is_in(actionable_types))
        
        if 'probability' in df.columns:
            df = df.filter(pl.col('probability') >= self.min_probability)
            df = df.sort('probability', descending=True)
        
        for row in df.iter_rows(named=True):
            if total_open >= self.max_total_positions:
                break
            
            if remaining_allocation <= self.min_order_notional:
                break
            
            if available_cash <= self.min_order_notional:
                break
            
            pair = row[self.pair_col]
            signal_type = row['signal_type']
            probability = row.get('probability', 1.0) or 1.0
            
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
            
            notional = min(notional, available_cash * 0.99) 
            notional = min(notional, remaining_allocation)
            
            if notional < self.min_order_notional:
                continue
            
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
                    'requested_notional': notional,
                }
            )
            orders.append(order)
            
            total_open += 1
            available_cash -= notional * 1.002  
            remaining_allocation -= notional
            positions_by_pair.setdefault(pair, []).append(None)  
        
        return orders

