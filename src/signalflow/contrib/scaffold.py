"""Component scaffolding, validation, and testing helpers.

Generates boilerplate code for custom SignalFlow components,
validates existing component classes, and runs quick sanity checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from signalflow.core import SfComponentType

# ── Type mapping ───────────────────────────────────────────────────────

_TYPE_MAP: dict[str, SfComponentType] = {
    "detector": SfComponentType.DETECTOR,
    "feature": SfComponentType.FEATURE,
    "validator": SfComponentType.VALIDATOR,
    "labeler": SfComponentType.LABELER,
    "entry": SfComponentType.STRATEGY_ENTRY_RULE,
    "entry_rule": SfComponentType.STRATEGY_ENTRY_RULE,
    "exit": SfComponentType.STRATEGY_EXIT_RULE,
    "exit_rule": SfComponentType.STRATEGY_EXIT_RULE,
    "signal_feature": SfComponentType.SIGNAL_FEATURE,
    "data_source": SfComponentType.RAW_DATA_SOURCE,
    "strategy_metric": SfComponentType.STRATEGY_METRIC,
    "signal_metric": SfComponentType.SIGNAL_METRIC,
}

# ── Templates ──────────────────────────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    "detector": '''"""Custom signal detector: {name}."""

from __future__ import annotations

import polars as pl
import signalflow as sf
from signalflow.core import Signals
from signalflow.detector.base import SignalDetector


@sf.detector("{registry_name}")
class {class_name}(SignalDetector):
    """{name} signal detector.

    Emits "rise" / "fall" signals based on custom logic.
    """

    # Add configurable parameters as dataclass fields:
    # period: int = 14
    # threshold: float = 0.5

    def detect(self, features: pl.DataFrame, context: dict | None = None) -> Signals:
        """Detect signals from feature DataFrame.

        Args:
            features: DataFrame with OHLCV + computed features per pair.
            context: Optional context dict.

        Returns:
            Signals object with detected entry/exit signals.
        """
        signals = Signals()

        for pair in features[self.pair_col].unique().to_list():
            pair_df = features.filter(pl.col(self.pair_col) == pair)

            # TODO: Implement your detection logic here
            # Example: emit "rise" when condition is met
            # for row in pair_df.iter_rows(named=True):
            #     if row["close"] > row["sma_50"]:
            #         signals.add(pair, row["timestamp"], "rise")

            _ = pair_df  # remove when implemented

        return signals
''',
    "feature": '''"""Custom feature: {name}."""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import signalflow as sf
from signalflow.feature.base import Feature


@sf.feature("{registry_name}")
class {class_name}(Feature):
    """{name} feature.

    Computes custom feature columns from OHLCV data.
    """

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["{snake_name}"]

    # Add configurable parameters:
    # period: int = 14

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute feature for a single pair.

        Args:
            df: DataFrame with OHLCV data for one pair.

        Returns:
            DataFrame with added feature column(s).
        """
        # TODO: Implement your feature computation
        return df.with_columns(
            pl.lit(0.0).alias("{snake_name}")
        )
''',
    "validator": '''"""Custom signal validator: {name}."""

from __future__ import annotations

from typing import Any

import polars as pl
import signalflow as sf
from signalflow.core import Signals
from signalflow.validator.base import SignalValidator


@sf.validator("{registry_name}")
class {class_name}(SignalValidator):
    """{name} signal validator.

    Filters signals using a trained ML model.
    """

    # model_type: str = "lightgbm"
    # model_params: dict = field(default_factory=dict)

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> {class_name}:
        """Train the validation model."""
        # TODO: Implement training logic
        return self

    def predict(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Filter signals using trained model."""
        # TODO: Implement prediction logic
        return signals

    def predict_proba(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Attach probability scores to signals."""
        # TODO: Implement probability prediction
        return signals

    def save(self, path: str) -> None:
        """Save model to disk."""

    @classmethod
    def load(cls, path: str) -> {class_name}:
        """Load model from disk."""
        return cls()
''',
    "labeler": '''"""Custom labeler: {name}."""

from __future__ import annotations

import polars as pl
import signalflow as sf
from signalflow.target.base import Labeler


@sf.labeler("{registry_name}")
class {class_name}(Labeler):
    """{name} labeler.

    Computes target labels for signal evaluation.
    """

    # Add configurable parameters:
    # horizon: int = 10
    # threshold: float = 0.02

    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict | None = None,
    ) -> pl.DataFrame:
        """Compute labels for a single pair group.

        MUST preserve row count (return same number of rows as input).

        Args:
            group_df: DataFrame for a single pair.
            data_context: Optional context.

        Returns:
            DataFrame with added label column.
        """
        # TODO: Implement labeling logic
        return group_df.with_columns(
            pl.lit(0).alias(self.out_col)
        )
''',
    "entry": '''"""Custom entry rule: {name}."""

from __future__ import annotations

import signalflow as sf
from signalflow.core import Order, Signals, StrategyState
from signalflow.strategy.component.base import EntryRule


@sf.entry("{registry_name}")
class {class_name}(EntryRule):
    """{name} entry rule.

    Generates entry orders based on signals and current state.
    """

    # Add configurable parameters:
    # size_pct: float = 0.1

    def check_entries(
        self,
        signals: Signals,
        prices: dict[str, float],
        state: StrategyState,
    ) -> list[Order]:
        """Check signals and generate entry orders.

        Args:
            signals: Current signals.
            prices: Latest prices per pair.
            state: Current strategy state.

        Returns:
            List of entry orders.
        """
        orders: list[Order] = []
        # TODO: Implement entry logic
        return orders
''',
    "exit": '''"""Custom exit rule: {name}."""

from __future__ import annotations

import signalflow as sf
from signalflow.core import Order, Position, StrategyState
from signalflow.strategy.component.base import ExitRule


@sf.exit("{registry_name}")
class {class_name}(ExitRule):
    """{name} exit rule.

    Generates exit orders based on positions and current prices.
    """

    # Add configurable parameters:
    # tp: float = 0.03
    # sl: float = 0.01

    def check_exits(
        self,
        positions: list[Position],
        prices: dict[str, float],
        state: StrategyState,
    ) -> list[Order]:
        """Check positions and generate exit orders.

        Args:
            positions: Current open positions.
            prices: Latest prices per pair.
            state: Current strategy state.

        Returns:
            List of exit orders.
        """
        orders: list[Order] = []
        # TODO: Implement exit logic
        return orders
''',
    "signal_feature": '''"""Custom signal feature: {name}."""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import signalflow as sf
from signalflow.signal_feature.base import SignalFeature


@sf.signal_feature("{registry_name}")
class {class_name}(SignalFeature):
    """{name} signal feature.

    Computes meta-features from signal history.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["{snake_name}"]

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict | None = None,
    ) -> pl.DataFrame:
        """Compute signal feature.

        MUST return same (pair, timestamp) rows as input.

        Args:
            signals: Signal history DataFrame.
            labels: Optional resolved labels (if requires_labels=True).
            context: Optional context.

        Returns:
            DataFrame with added feature column(s).
        """
        # TODO: Implement signal feature computation
        return signals.with_columns(
            pl.lit(0.0).alias("{snake_name}")
        )
''',
}

# ── Public API ─────────────────────────────────────────────────────────


def scaffold(
    name: str,
    *,
    component_type: str,
    output_dir: str | Path = ".",
) -> Path:
    """Generate boilerplate code for a custom SignalFlow component.

    Args:
        name: Component name (e.g. "my_rsi_detector").
        component_type: One of "detector", "feature", "validator",
            "labeler", "entry", "exit", "signal_feature".
        output_dir: Directory to write the generated file.

    Returns:
        Path to the generated file.

    Raises:
        ValueError: If component_type is not supported.
    """
    ct = component_type.lower().strip()
    if ct not in _TEMPLATES:
        supported = ", ".join(sorted(_TEMPLATES.keys()))
        msg = f"Unsupported component_type {ct!r}. Supported: {supported}"
        raise ValueError(msg)

    snake = name.lower().replace("-", "_").replace(" ", "_")
    class_name = "".join(word.capitalize() for word in snake.split("_"))
    registry_name = f"custom/{snake}"

    code = _TEMPLATES[ct].format(
        name=name,
        class_name=class_name,
        registry_name=registry_name,
        snake_name=snake,
    )

    out = Path(output_dir) / f"{snake}.py"
    out.write_text(code, encoding="utf-8")
    return out


def validate_component(cls: type[Any]) -> list[str]:
    """Validate that a class is a proper SignalFlow component.

    Checks:
    - Has ``component_type`` attribute (ClassVar).
    - Component type is a valid :class:`SfComponentType`.
    - Has required methods for its component type.

    Args:
        cls: The component class to validate.

    Returns:
        List of warning/error messages. Empty list = valid.
    """
    issues: list[str] = []

    # Check component_type
    if not hasattr(cls, "component_type"):
        issues.append("Missing 'component_type' ClassVar")
        return issues

    ct = cls.component_type
    if not isinstance(ct, SfComponentType):
        issues.append(f"component_type is {type(ct).__name__}, expected SfComponentType")
        return issues

    # Check required methods per type
    required: dict[SfComponentType, list[str]] = {
        SfComponentType.DETECTOR: ["detect"],
        SfComponentType.FEATURE: ["compute_pair"],
        SfComponentType.VALIDATOR: ["fit", "predict"],
        SfComponentType.LABELER: ["compute_group"],
        SfComponentType.STRATEGY_ENTRY_RULE: ["check_entries"],
        SfComponentType.STRATEGY_EXIT_RULE: ["check_exits"],
        SfComponentType.SIGNAL_FEATURE: ["compute"],
    }

    for method in required.get(ct, []):
        if not hasattr(cls, method):
            issues.append(f"Missing required method: {method}")
        elif not callable(getattr(cls, method)):
            issues.append(f"'{method}' is not callable")

    return issues


def check_component(cls: type[Any], **kwargs: Any) -> dict[str, Any]:
    """Quick sanity check for a component class.

    Validates the class, instantiates it, and checks basic properties.

    Args:
        cls: The component class to test.
        **kwargs: Arguments passed to the constructor.

    Returns:
        Dict with "valid" (bool), "issues" (list), "instance" (object or None).
    """
    result: dict[str, Any] = {"valid": False, "issues": [], "instance": None}

    # Validate class
    issues = validate_component(cls)
    if issues:
        result["issues"] = issues
        return result

    # Try instantiation
    try:
        instance = cls(**kwargs)
        result["instance"] = instance
    except Exception as e:
        result["issues"].append(f"Instantiation failed: {e}")
        return result

    # Check component_type on instance
    if not hasattr(instance, "component_type"):
        result["issues"].append("Instance missing component_type after init")
        return result

    result["valid"] = True
    return result
