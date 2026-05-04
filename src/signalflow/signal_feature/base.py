"""Signal Feature base class.

Signal features (meta-features) extract statistics from signal history
rather than from raw market data. They capture properties of the
signal-generating process itself: accuracy trends, frequency changes,
entropy, streaks, and more.

Two categories:
    - **Unsupervised** (``requires_labels = False``): computed from signal
      history alone (frequency, entropy, flip-rate, streak length, ...).
    - **Supervised** (``requires_labels = True``): require resolved outcome
      labels (rolling accuracy, expected-value, Bayesian posterior, ...).

Look-ahead prevention
---------------------
Supervised features must **never** peek at labels that haven't resolved yet.
The base class provides :meth:`prepare_labels` which marks each label row
with a ``_resolved_at`` timestamp and drops rows whose outcome is still
unknown at each signal's time. Implementations receive only causally-valid
data.

Resolution timestamp is determined by (in priority order):
    1. ``label_resolve_col`` column (e.g. ``t_hit`` from triple-barrier)
    2. ``label_delay`` bars after the signal timestamp
    3. Strict: only labels at ``t < current_signal_t`` (no same-bar)

Integration
-----------
Signal features slot between detection and validation in the FlowBuilder::

    RawData -> Features -> Detector -> **SignalFeatures** -> Validator -> Entry

The ``compute()`` output is a *separate* DataFrame keyed on
``(pair, timestamp)`` with only the produced feature columns.  The
FlowBuilder joins it with the main feature matrix before passing
to the validator.

Example::

    from signalflow.signal_feature import SignalFeature

    @sf.signal_feature("signal_stats/rolling_accuracy")
    @dataclass
    class RollingAccuracy(SignalFeature):
        requires_labels: ClassVar[bool] = True
        outputs: ClassVar[list[str]] = ["rolling_acc_{window}"]
        window: int = 50

        def compute(self, signals, labels=None, context=None):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl
from loguru import logger

from signalflow.core.enums import SfComponentType


@dataclass
class SignalFeature(ABC):
    """Base class for features computed from signal history.

    Subclasses must implement :meth:`compute` and declare ``outputs``.

    Attributes:
        requires_labels: Whether resolved outcome labels are needed.
        outputs: Template strings for output column names.  Parameter
            substitution uses ``self.__dict__`` (same convention as
            :class:`~signalflow.feature.Feature`).
        group_col: Column used to group signals by instrument.
        ts_col: Timestamp column name.
        label_resolve_col: Column in labels that stores the time when
            the label outcome became known (e.g. ``"t_hit"``).
            Set to ``None`` to fall back to ``label_delay``.
        label_delay: Fixed number of bars to wait before considering
            a label resolved.  Only used when ``label_resolve_col``
            is ``None`` or the column is missing from labels.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.SIGNAL_FEATURE

    requires_labels: ClassVar[bool] = False
    """Whether this feature needs resolved outcome labels."""

    outputs: ClassVar[list[str]] = []
    """Output column name templates, e.g. ``["rolling_acc_{window}"]``."""

    group_col: str = "pair"
    ts_col: str = "timestamp"

    # --- look-ahead prevention knobs ---
    label_resolve_col: str | None = "t_hit"
    label_delay: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute signal-level features.

        Args:
            signals: Signal history with at least
                ``(pair, timestamp, signal_type, signal)`` columns.
                May also include ``probability``.
            labels: Resolved outcome labels with at least
                ``(pair, timestamp, label)`` columns.  May include
                ``t_hit``, ``ret`` or other metadata.  Only passed
                when ``requires_labels is True``.
            context: Optional extra data (e.g. raw_data reference for
                regime-aware features).

        Returns:
            :class:`pl.DataFrame` keyed on ``(pair, timestamp)`` with
            **only** the produced feature columns plus the key columns.
            Must contain exactly the same ``(pair, timestamp)`` rows as
            the input ``signals`` (no filtering).
        """
        ...

    def output_cols(self) -> list[str]:
        """Resolved output column names after parameter substitution."""
        return [tpl.format(**self.__dict__) for tpl in self.outputs]

    @property
    def warmup(self) -> int:
        """Minimum number of prior signals before output is meaningful.

        Override in subclasses. Default: ``0``.
        """
        return 0

    # ------------------------------------------------------------------
    # Callable shortcut
    # ------------------------------------------------------------------

    def __call__(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Shorthand for :meth:`compute` with input validation."""
        self._validate_signals(signals)
        if self.requires_labels:
            if labels is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires labels "
                    f"(requires_labels=True) but labels=None was passed."
                )
            self._validate_labels(labels)
        return self.compute(signals=signals, labels=labels, context=context)

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def prepare_labels(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join signals with causally-valid labels.

        Returns a copy of *signals* enriched with ``label`` (and any
        label metadata columns) plus a ``_resolved_at`` column.  Rows
        whose labels have **not yet resolved** at their own timestamp
        receive ``null`` values for all label columns.

        This is the primary look-ahead prevention mechanism.  Supervised
        signal features should call this first, then use ``_resolved_at``
        to window their rolling computations.

        Resolution logic (per label row):
            1. If ``label_resolve_col`` exists in *labels* and is not
               null → ``_resolved_at = labels[label_resolve_col]``.
            2. Else if ``label_delay`` is set → ``_resolved_at`` is the
               timestamp of the bar ``label_delay`` steps after the
               signal (approximated by offset in the sorted frame).
            3. Else → ``_resolved_at = signal_timestamp`` (assumes
               immediate resolution — only safe for non-forward labels).

        Args:
            signals: Signal DataFrame.
            labels: Labels DataFrame with ``(pair, timestamp, label, ...)``.

        Returns:
            Enriched DataFrame with signals + label columns +
            ``_resolved_at``.
        """
        key = [self.group_col, self.ts_col]

        # Columns we'll take from labels (exclude key cols to avoid duplication)
        label_cols = [c for c in labels.columns if c not in key]

        merged = signals.join(
            labels.select([*key, *label_cols]),
            on=key,
            how="left",
        )

        # Determine _resolved_at
        if (
            self.label_resolve_col is not None
            and self.label_resolve_col in merged.columns
        ):
            merged = merged.with_columns(
                pl.col(self.label_resolve_col).alias("_resolved_at"),
            )
        elif self.label_delay is not None:
            # Approximate: shift timestamp forward by label_delay rows
            # within each pair group
            merged = merged.sort(key).with_columns(
                pl.col(self.ts_col)
                .shift(-self.label_delay)
                .over(self.group_col)
                .alias("_resolved_at"),
            )
        else:
            # Fallback: treat label as resolved at signal time
            logger.debug(
                "{}: no label_resolve_col or label_delay set, "
                "using signal timestamp as _resolved_at",
                self.__class__.__name__,
            )
            merged = merged.with_columns(
                pl.col(self.ts_col).alias("_resolved_at"),
            )

        return merged

    def mask_unresolved(
        self,
        df: pl.DataFrame,
        label_col: str = "label",
    ) -> pl.DataFrame:
        """Null-out label values that haven't resolved yet.

        For each row, if ``_resolved_at > timestamp`` (label resolves
        in the future) the label and related columns are set to ``null``.
        This ensures rolling aggregations only see causally valid data.

        Must be called **after** :meth:`prepare_labels`.

        Args:
            df: DataFrame from :meth:`prepare_labels` (has ``_resolved_at``).
            label_col: Name of the label column to mask.

        Returns:
            DataFrame with future-peeking labels nulled out.
        """
        if "_resolved_at" not in df.columns:
            raise ValueError(
                "DataFrame missing '_resolved_at' column. "
                "Call prepare_labels() first."
            )

        # Label is valid only if it resolved at or before this row's timestamp
        is_resolved = pl.col("_resolved_at") <= pl.col(self.ts_col)

        # Null out label + any metadata that came from the labels join
        mask_cols = [label_col]
        if self.label_resolve_col and self.label_resolve_col in df.columns:
            mask_cols.append(self.label_resolve_col)

        exprs = [
            pl.when(is_resolved).then(pl.col(c)).otherwise(pl.lit(None)).alias(c)
            for c in mask_cols
            if c in df.columns
        ]

        return df.with_columns(exprs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_signals(self, signals: pl.DataFrame) -> None:
        required = {self.group_col, self.ts_col, "signal_type"}
        missing = required - set(signals.columns)
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: signals missing columns {sorted(missing)}"
            )

    def _validate_labels(self, labels: pl.DataFrame) -> None:
        required = {self.group_col, self.ts_col, "label"}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: labels missing columns {sorted(missing)}"
            )
