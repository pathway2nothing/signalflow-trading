"""Base classes for position sizing strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from signalflow.core import SfComponentType

if TYPE_CHECKING:
    from signalflow.core import StrategyState


@dataclass
class SignalContext:
    """Context for a single signal being sized.

    Provides all relevant information about a signal for sizing decisions.

    Attributes:
        pair: Trading pair (e.g., "BTCUSDT").
        signal_type: Signal direction ("rise", "fall", "none").
        probability: Signal confidence [0, 1].
        price: Current market price.
        timestamp: Signal timestamp.
        meta: Additional signal metadata from detector.
    """

    pair: str
    signal_type: str
    probability: float
    price: float
    timestamp: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSizer(ABC):
    """Base class for position sizing strategies.

    Computes the notional value (in quote currency) for a trade based on
    signal strength, portfolio state, and market conditions.

    Design principles:
        - Sizers compute NOTIONAL value, not quantity
        - Quantity = notional / price (computed by entry rule)
        - Sizers should be stateless where possible
        - Historical data accessed via state.runtime or state.metrics

    Example:
        >>> sizer = FixedFractionSizer(fraction=0.02)
        >>> notional = sizer.compute_size(signal_ctx, state, prices)
        >>> qty = notional / prices[signal_ctx.pair]
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.STRATEGY_ENTRY_RULE

    @abstractmethod
    def compute_size(
        self,
        signal: SignalContext,
        state: StrategyState,
        prices: dict[str, float],
    ) -> float:
        """Compute position size (notional value) for a signal.

        Args:
            signal: Context about the signal being sized.
            state: Current strategy state (portfolio, metrics, runtime).
            prices: Current prices for all pairs.

        Returns:
            Notional value in quote currency (e.g., USDT).
            Return 0.0 to skip this signal.
        """
        ...
