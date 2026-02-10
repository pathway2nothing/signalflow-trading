"""Registry of known signal values per category.

Provides advisory registries of known signal types and their mappings.
The system does NOT require values to be registered -- new signal types
can be used without modifying this module. This is purely for discovery,
validation helpers, and directional mapping.
"""

from __future__ import annotations

from signalflow.core.enums import SignalCategory

# Known signal values per category (advisory, not enforced)
KNOWN_SIGNALS: dict[str, set[str]] = {
    SignalCategory.PRICE_DIRECTION.value: {"rise", "fall", "flat"},
    SignalCategory.PRICE_STRUCTURE.value: {
        "local_top",
        "local_bottom",
        "breakout_up",
        "breakout_down",
        "range_bound",
        "higher_high",
        "lower_low",
    },
    SignalCategory.TREND_MOMENTUM.value: {
        "trend_start",
        "trend_continuation",
        "trend_reversal",
        "trend_exhaustion",
        "oversold",
        "overbought",
    },
    SignalCategory.VOLATILITY.value: {
        "vol_high",
        "vol_low",
        "vol_expansion",
        "vol_contraction",
        "vol_regime_shift",
    },
    SignalCategory.VOLUME_LIQUIDITY.value: {
        "volume_spike",
        "volume_drought",
        "volume_divergence",
        "accumulation",
        "distribution",
    },
    SignalCategory.MARKET_WIDE.value: {
        "market_crash",
        "market_rally",
        "correlation_spike",
        "decorrelation",
        "sector_divergence",
        "regime_shift",
    },
    SignalCategory.ANOMALY.value: {
        "black_swan",
        "flash_crash",
        "manipulation",
    },
}

# Mapping: signal_type -> order side for directional trading
DIRECTIONAL_SIGNAL_MAP: dict[str, str] = {
    "rise": "BUY",
    "fall": "SELL",
    "local_bottom": "BUY",
    "local_top": "SELL",
    "breakout_up": "BUY",
    "breakout_down": "SELL",
    "oversold": "BUY",
    "overbought": "SELL",
}


def get_known_signals(category: str | SignalCategory) -> set[str]:
    """Get known signal values for a category.

    Args:
        category: Category name or SignalCategory enum.

    Returns:
        Set of known signal type strings. Empty set if category unknown.
    """
    key = category.value if isinstance(category, SignalCategory) else category
    return KNOWN_SIGNALS.get(key, set())


def get_all_known_signals() -> set[str]:
    """Get all known signal values across all categories."""
    result: set[str] = set()
    for signals in KNOWN_SIGNALS.values():
        result |= signals
    return result


def get_directional_side(signal_type: str) -> str | None:
    """Get order side (BUY/SELL) for a signal type.

    Args:
        signal_type: Signal type string.

    Returns:
        "BUY", "SELL", or None if signal has no directional mapping.
    """
    return DIRECTIONAL_SIGNAL_MAP.get(signal_type)
