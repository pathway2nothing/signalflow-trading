"""Entry rules, filters, and signal aggregation."""

from signalflow.strategy.component.entry.aggregation import SignalAggregator, VotingMode
from signalflow.strategy.component.entry.filters import (
    CompositeEntryFilter,
    CorrelationFilter,
    DrawdownFilter,
    EntryFilter,
    PriceDistanceFilter,
    RegimeFilter,
    SignalAccuracyFilter,
    TimeOfDayFilter,
    VolatilityFilter,
)
from signalflow.strategy.component.entry.fixed_size import FixedSizeEntryRule
from signalflow.strategy.component.entry.signal import SignalEntryRule

__all__ = [
    # Filters
    "CompositeEntryFilter",
    "CorrelationFilter",
    "DrawdownFilter",
    "EntryFilter",
    # Entry rules
    "FixedSizeEntryRule",
    "PriceDistanceFilter",
    "RegimeFilter",
    "SignalAccuracyFilter",
    # Aggregation
    "SignalAggregator",
    "SignalEntryRule",
    "TimeOfDayFilter",
    "VolatilityFilter",
    "VotingMode",
]
