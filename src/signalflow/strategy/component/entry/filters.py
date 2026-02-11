"""Entry filters for pre-trade validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from signalflow.core import SfComponentType, sf_component
from signalflow.strategy.component.sizing.base import SignalContext

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
class EntryFilter(ABC):
    """Base class for entry filters.

    Filters determine whether a signal should be acted upon.
    All filters must pass (AND logic) for entry to proceed.

    Design principles:
        - Filters are binary (allow/reject)
        - Should provide rejection reason for debugging
        - Can be composed via CompositeEntryFilter
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ENTRY_RULE

    @abstractmethod
    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        """Check if entry is allowed.

        Args:
            signal: Signal context.
            state: Strategy state.
            prices: Current prices.

        Returns:
            Tuple of (allowed, reason).
            reason is empty string if allowed, else rejection reason.
        """
        ...


@dataclass
@sf_component(name="composite_entry_filter")
class CompositeEntryFilter(EntryFilter):
    """Combines multiple entry filters.

    Args:
        filters: List of filters to apply.
        require_all: If True (default), all must pass. If False, any can pass.

    Example:
        >>> composite = CompositeEntryFilter(
        ...     filters=[DrawdownFilter(max_drawdown=0.10), VolatilityFilter()],
        ...     require_all=True
        ... )
    """

    filters: list[EntryFilter] = field(default_factory=list)
    require_all: bool = True

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        if not self.filters:
            return True, ""

        reasons = []
        passes = []

        for f in self.filters:
            allowed, reason = f.allow_entry(signal, state, prices)
            passes.append(allowed)
            if not allowed:
                reasons.append(f"{f.__class__.__name__}: {reason}")

        if self.require_all:
            if all(passes):
                return True, ""
            return False, "; ".join(reasons)
        else:
            if any(passes):
                return True, ""
            return False, "; ".join(reasons)


@dataclass
@sf_component(name="regime_filter")
class RegimeFilter(EntryFilter):
    """Filter entries based on market regime.

    Only allow entries when market regime matches signal type:
    - Bullish signals in trend-up or mean-reversion-oversold regimes
    - Bearish signals in trend-down or mean-reversion-overbought regimes

    Regime detected via state.runtime["regime"][pair] or global regime.

    Args:
        signal_regime_map: Mapping signal_type -> "bullish"/"bearish".
            When set, overrides legacy "rise"/"fall" hardcoding.
            None = legacy behavior (only "rise" and "fall" are regime-checked).
        regime_key: Key in state.runtime for regime data.
        allowed_regimes_bullish: Regimes allowing bullish entries.
        allowed_regimes_bearish: Regimes allowing bearish entries.
    """

    signal_regime_map: dict[str, str] | None = None  # signal_type -> "bullish"/"bearish"

    regime_key: str = "regime"
    allowed_regimes_bullish: list[str] = field(default_factory=lambda: ["trend_up", "mean_reversion_oversold"])
    allowed_regimes_bearish: list[str] = field(default_factory=lambda: ["trend_down", "mean_reversion_overbought"])

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        regime_data = state.runtime.get(self.regime_key, {})

        # Try pair-specific regime, then global
        regime = regime_data.get(signal.pair) or regime_data.get("global")

        if regime is None:
            return True, ""  # No regime data, allow

        # Determine regime category for this signal_type
        if self.signal_regime_map is not None:
            category = self.signal_regime_map.get(signal.signal_type)
        else:
            # Legacy behavior
            category = {"rise": "bullish", "fall": "bearish"}.get(signal.signal_type)

        if category is None:
            return True, ""  # Unknown signal type, allow

        if category == "bullish":
            if regime in self.allowed_regimes_bullish:
                return True, ""
            return False, f"regime={regime} not in {self.allowed_regimes_bullish}"

        if category == "bearish":
            if regime in self.allowed_regimes_bearish:
                return True, ""
            return False, f"regime={regime} not in {self.allowed_regimes_bearish}"

        return True, ""


@dataclass
@sf_component(name="volatility_filter")
class VolatilityFilter(EntryFilter):
    """Skip entries in extreme volatility conditions.

    Args:
        volatility_key: Key in state.runtime for volatility data (default: "atr").
        min_volatility: Minimum relative volatility to allow entry.
        max_volatility: Maximum relative volatility to allow entry.
        use_relative: If True, compare vol/price ratio instead of absolute.
    """

    volatility_key: str = "atr"
    min_volatility: float = 0.0
    max_volatility: float = float("inf")
    use_relative: bool = True

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        vol_data = state.runtime.get(self.volatility_key, {})
        vol = vol_data.get(signal.pair)

        if vol is None:
            return True, ""  # No data, allow

        if self.use_relative and signal.price > 0:
            vol = vol / signal.price

        if vol < self.min_volatility:
            return False, f"volatility={vol:.4f} < min={self.min_volatility}"
        if vol > self.max_volatility:
            return False, f"volatility={vol:.4f} > max={self.max_volatility}"

        return True, ""


@dataclass
@sf_component(name="drawdown_filter")
class DrawdownFilter(EntryFilter):
    """Pause trading after significant drawdown.

    Args:
        max_drawdown: Maximum drawdown before pausing (e.g., 0.10 = 10%).
        recovery_threshold: Resume when drawdown reduces to this level.
        drawdown_key: Key in state.metrics for current drawdown.
    """

    max_drawdown: float = 0.10
    recovery_threshold: float = 0.05
    drawdown_key: str = "current_drawdown"
    _paused: bool = field(default=False, repr=False)

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        current_dd = state.metrics.get(self.drawdown_key, 0.0)

        if current_dd >= self.max_drawdown:
            self._paused = True
            return False, f"drawdown={current_dd:.2%} >= max={self.max_drawdown:.2%}"

        if self._paused and current_dd > self.recovery_threshold:
            return False, f"paused, drawdown={current_dd:.2%} > recovery={self.recovery_threshold:.2%}"

        self._paused = False
        return True, ""


@dataclass
@sf_component(name="correlation_filter")
class CorrelationFilter(EntryFilter):
    """Avoid concentrated positions in correlated assets.

    Rejects entry if already holding highly correlated assets.

    Args:
        correlation_key: Key in state.runtime for correlation matrix.
        max_correlation: Maximum allowed correlation with existing positions.
        max_correlated_positions: Max positions in correlated group.
    """

    correlation_key: str = "correlations"
    max_correlation: float = 0.7
    max_correlated_positions: int = 2

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        corr_matrix = state.runtime.get(self.correlation_key, {})

        if not corr_matrix:
            return True, ""  # No correlation data

        open_pairs = {p.pair for p in state.portfolio.open_positions()}

        correlated_count = 0
        for existing_pair in open_pairs:
            pair_key = tuple(sorted([signal.pair, existing_pair]))
            corr = corr_matrix.get(pair_key, 0.0)

            if abs(corr) >= self.max_correlation:
                correlated_count += 1

        if correlated_count >= self.max_correlated_positions:
            return False, f"already holding {correlated_count} correlated positions"

        return True, ""


@dataclass
@sf_component(name="time_of_day_filter")
class TimeOfDayFilter(EntryFilter):
    """Restrict trading to specific hours.

    Args:
        allowed_hours: List of hours (0-23) when trading is allowed.
        blocked_hours: List of hours (0-23) when trading is blocked.

    Note: If both are None, all hours are allowed.
    """

    allowed_hours: list[int] | None = None
    blocked_hours: list[int] | None = None

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        if signal.timestamp is None:
            return True, ""

        hour = signal.timestamp.hour

        if self.blocked_hours and hour in self.blocked_hours:
            return False, f"hour={hour} is blocked"

        if self.allowed_hours and hour not in self.allowed_hours:
            return False, f"hour={hour} not in allowed hours"

        return True, ""


@dataclass
@sf_component(name="price_distance_filter")
class PriceDistanceFilter(EntryFilter):
    """Filter entries based on price distance from existing positions.

    For grid strategies: prevents buying when price is too close to
    existing positions in the same pair.

    Args:
        signal_direction_map: Mapping signal_type -> "long"/"short".
            When set, overrides legacy "rise"/"fall" hardcoding.
            None = legacy behavior (only "rise" and "fall" are direction-aware).
        min_distance_pct: Minimum price difference as percentage (e.g., 0.02 = 2%).
        direction_aware: If True, check distance based on position direction.
            - LONG: new entry must be below existing entry by min_distance_pct
            - SHORT: new entry must be above existing entry by min_distance_pct
            If False, check absolute distance in either direction.

    Example:
        >>> # Grid strategy: only buy when price drops 2% from last position
        >>> filter = PriceDistanceFilter(min_distance_pct=0.02, direction_aware=True)
    """

    signal_direction_map: dict[str, str] | None = None  # signal_type -> "long"/"short"

    min_distance_pct: float = 0.02
    direction_aware: bool = True

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        # Get existing positions in the same pair
        pair_positions = [p for p in state.portfolio.open_positions() if p.pair == signal.pair]

        if not pair_positions:
            return True, ""  # No existing positions

        # Get the most recent position's entry price
        # (could also check all positions or closest price)
        last_position = max(pair_positions, key=lambda p: p.entry_time or p.id)
        last_entry_price = last_position.entry_price

        if last_entry_price <= 0 or signal.price <= 0:
            return True, ""

        price_change_pct = (signal.price - last_entry_price) / last_entry_price

        if self.direction_aware:
            # Determine direction for this signal_type
            if self.signal_direction_map is not None:
                direction = self.signal_direction_map.get(signal.signal_type)
            else:
                # Legacy behavior
                direction = {"rise": "long", "fall": "short"}.get(signal.signal_type)

            if direction == "long":
                # LONG: we want to buy lower (DCA)
                if price_change_pct > -self.min_distance_pct:
                    return False, (
                        f"price too close to last entry: {price_change_pct:.2%} > -{self.min_distance_pct:.2%}"
                    )
            elif direction == "short":
                # SHORT: we want to sell higher
                if price_change_pct < self.min_distance_pct:
                    return False, (
                        f"price too close to last entry: {price_change_pct:.2%} < {self.min_distance_pct:.2%}"
                    )
        else:
            # Check absolute distance in either direction
            if abs(price_change_pct) < self.min_distance_pct:
                return False, (f"price too close to last entry: |{price_change_pct:.2%}| < {self.min_distance_pct:.2%}")

        return True, ""


@dataclass
@sf_component(name="signal_accuracy_filter")
class SignalAccuracyFilter(EntryFilter):
    """Filter based on real-time signal accuracy metrics.

    Tracks detector/model accuracy and pauses trading when accuracy drops.
    Useful for detecting model degradation or regime changes.

    Args:
        accuracy_key: Key in state.runtime for accuracy data.
        min_accuracy: Minimum required accuracy to allow entry.
        min_samples: Minimum samples before applying filter.
        window_key: Optional key for accuracy over recent window only.

    Example:
        >>> # Pause if recent signal accuracy drops below 45%
        >>> filter = SignalAccuracyFilter(min_accuracy=0.45, min_samples=20)
    """

    accuracy_key: str = "signal_accuracy"
    min_accuracy: float = 0.45
    min_samples: int = 20
    window_key: str | None = None

    def allow_entry(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        accuracy_data = state.runtime.get(self.accuracy_key, {})

        # Try pair-specific accuracy, then global
        key = self.window_key or "overall"
        pair_accuracy = accuracy_data.get(signal.pair, {})
        accuracy = pair_accuracy.get(key) or accuracy_data.get(key)
        samples = pair_accuracy.get("samples", 0) or accuracy_data.get("samples", 0)

        if accuracy is None or samples < self.min_samples:
            return True, ""  # Insufficient data, allow

        if accuracy < self.min_accuracy:
            return False, f"signal_accuracy={accuracy:.2%} < min={self.min_accuracy:.2%}"

        return True, ""
