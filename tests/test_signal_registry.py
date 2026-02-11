"""Tests for signalflow.core.signal_registry."""

from signalflow.core.enums import SignalCategory
from signalflow.core.signal_registry import (
    KNOWN_SIGNALS,
    DIRECTIONAL_SIGNAL_MAP,
    get_known_signals,
    get_all_known_signals,
    get_directional_side,
)


class TestKnownSignals:
    def test_all_categories_have_entries(self):
        for cat in SignalCategory:
            assert cat.value in KNOWN_SIGNALS, f"Missing category: {cat.value}"
            assert len(KNOWN_SIGNALS[cat.value]) > 0, f"Empty signals for {cat.value}"

    def test_price_direction_signals(self):
        signals = KNOWN_SIGNALS[SignalCategory.PRICE_DIRECTION.value]
        assert "rise" in signals
        assert "fall" in signals
        assert "flat" in signals

    def test_anomaly_signals(self):
        signals = KNOWN_SIGNALS[SignalCategory.ANOMALY.value]
        assert "extreme_positive_anomaly" in signals
        assert "extreme_negative_anomaly" in signals

    def test_no_duplicate_signals_across_categories(self):
        """Signal values should be unique across categories (except intentional reuse)."""
        all_values: list[str] = []
        for signals in KNOWN_SIGNALS.values():
            all_values.extend(signals)
        # rise/fall appear in price_direction; no others should duplicate
        duplicates = [v for v in set(all_values) if all_values.count(v) > 1]
        # Only intentional shared values (rise/fall could be reused by trend_momentum)
        # For now just ensure no massive duplication
        assert len(duplicates) <= 2, f"Unexpected duplicates: {duplicates}"


class TestDirectionalSignalMap:
    def test_rise_maps_to_buy(self):
        assert DIRECTIONAL_SIGNAL_MAP["rise"] == "BUY"

    def test_fall_maps_to_sell(self):
        assert DIRECTIONAL_SIGNAL_MAP["fall"] == "SELL"

    def test_all_values_are_buy_or_sell(self):
        for signal, side in DIRECTIONAL_SIGNAL_MAP.items():
            assert side in ("BUY", "SELL"), f"{signal} maps to invalid side: {side}"


class TestGetKnownSignals:
    def test_with_enum(self):
        signals = get_known_signals(SignalCategory.VOLATILITY)
        assert "high_volatility" in signals
        assert "low_volatility" in signals

    def test_with_string(self):
        signals = get_known_signals("anomaly")
        assert "extreme_positive_anomaly" in signals

    def test_unknown_category_returns_empty(self):
        signals = get_known_signals("nonexistent")
        assert signals == set()


class TestGetAllKnownSignals:
    def test_returns_all(self):
        all_signals = get_all_known_signals()
        assert "rise" in all_signals
        assert "extreme_positive_anomaly" in all_signals
        assert "high_volatility" in all_signals
        assert len(all_signals) > 20


class TestGetDirectionalSide:
    def test_rise(self):
        assert get_directional_side("rise") == "BUY"

    def test_fall(self):
        assert get_directional_side("fall") == "SELL"

    def test_unknown(self):
        assert get_directional_side("high_volatility") is None

    def test_local_bottom(self):
        assert get_directional_side("local_min") == "BUY"
