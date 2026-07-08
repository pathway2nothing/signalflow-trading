"""
Legacy ``Labeler`` base - ported to the ``Target`` API.

This is the restored legacy labeler base class. Every method of the original
(``compute`` / ``compute_group`` / ``compute_soft`` / ``compute_group_soft`` /
signal-masking / soft-label one-hot fallback) is preserved verbatim in spirit;
only the framework wiring changed:

* ``class Labeler(Target, ABC)`` - labelers ARE :class:`~signalflow.target.base.Target`
  specs, so they plug straight into ``ForecastModel(target=...)``.
* Column defaults follow the standard conventions: ``ts_col="ts"``, ``pair_col="pair"``.
* The 7-type component registry / ``SfComponentType`` / ``Signals`` container are
  gone; ``Signals`` is now a loose ``Any`` (signal masking still works when a
  signal frame is passed, but target use needs none).
* The :class:`Target` contract is implemented on the base:
    - :meth:`labels` runs ``compute`` and coerces the hard label column into a
      numeric ``label`` (``[pair, ts, label]``) for the classifier.
    - :attr:`horizon` reads the subclass's primary look-ahead field.

Subclasses register themselves with ``@register_target("<snake_case>")``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.enums import RawDataType, SignalCategory
from signalflow.enums import Signal as SignalType
from signalflow.errors import DegenerateTargetError
from signalflow.target.base import LABEL_COL, Target, resolve_bars

Signals = Any


_POSITIVE_LABELS: frozenset[str] = frozenset(
    {
        SignalType.RISE.value,
        "rise",
        "up",
        "take",
        "1",
        "true",
    }
)


_NEGATIVE_LABELS: frozenset[str] = frozenset(
    {
        SignalType.FALL.value,
        SignalType.NONE.value,
        "fall",
        "down",
        "skip",
        "flat",
        "neutral",
        "normal",
        "0",
        "false",
        "",
    }
)


@dataclass
class Labeler(Target, ABC):
    """Base class for Polars-only forward-looking signal labeling."""

    raw_data_type: RawDataType | str = RawDataType.SPOT

    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION
    """Signal category this labeler produces. Default: PRICE_DIRECTION."""

    soft_classes: ClassVar[tuple[str, ...]] = ()
    """Ordered class names emitted by :meth:`compute_soft` as ``p_<class>`` columns."""

    positive_classes: ClassVar[tuple[str, ...]] = ()
    """String labels mapped to the positive class (1.0); other non-null labels map to 0.0.

    Empty means the legacy coercion (anything not in the negative vocabulary becomes 1.0).
    Multi-class labelers must set this so their labels do not collapse to a constant.
    """

    soft_col_prefix: ClassVar[str] = "p_"
    """Prefix prepended to each soft class name to form the output column."""

    duration_fields: ClassVar[tuple[str, ...]] = ()
    """Field names whose value may be a duration string, resolved to bars at ``labels()`` time."""

    pair_col: str = "pair"
    ts_col: str = "ts"

    keep_input_columns: bool = False
    output_columns: list[str] | None = None
    filter_signal_type: SignalType | None = None

    mask_to_signals: bool = False
    out_col: str = "label"
    include_meta: bool = False
    meta_columns: tuple[str, ...] = ("t_hit", "ret")

    softness_k: float = 3.0
    """Sigmoid steepness for soft probability calibration."""

    @property
    def horizon(self) -> int:
        """Forward bars consumed - used for purge/embargo in the walk-forward CV."""

        own = self.__dict__.get("horizon")
        if isinstance(own, int) and own > 0:
            return own
        for attr in ("max_horizon", "max_bars", "n_bars", "lookforward", "max_lookforward", "flash_horizon", "window"):
            val = getattr(self, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        horizons = getattr(self, "horizons", None)
        if horizons:
            try:
                return int(max(horizons))
            except (TypeError, ValueError):
                pass
        return 1

    def _effective_horizon(self) -> int | str:
        """Raw forward-look value, keeping a duration string unresolved for ``horizon_bars``."""
        candidates = [self.__dict__.get("horizon")]
        for attr in ("max_horizon", "max_bars", "n_bars", "lookforward", "max_lookforward", "flash_horizon", "window"):
            candidates.append(getattr(self, attr, None))
        for val in candidates:
            if isinstance(val, str) and val:
                return val
            if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                return val
        return self.horizon

    def _resolve_durations(self, data: Dataset) -> "Labeler":
        """Return a copy with duration-string fields resolved to bar counts, or self if none."""
        import dataclasses

        overrides = {
            name: resolve_bars(getattr(self, name), data)
            for name in self.duration_fields
            if isinstance(getattr(self, name), str)
        }
        return dataclasses.replace(self, **overrides) if overrides else self

    def labels(self, data: Dataset, at: pl.DataFrame | None = None) -> pl.DataFrame:
        """Target contract: return numeric ``[pair, ts, label]``."""
        target = self._resolve_durations(data)
        computed = target.compute(data.frame)
        if target.out_col not in computed.columns:
            raise ValueError(f"{type(self).__name__}.labels: out_col {target.out_col!r} missing from compute() output")
        numeric = self._label_to_numeric(computed.get_column(target.out_col), target.positive_classes)
        distinct = sorted(numeric.drop_nulls().unique().to_list())
        if len(distinct) < 2:
            raise DegenerateTargetError(
                f"{type(self).__name__}.labels coerced to a degenerate target: non-null labels "
                f"collapse to {distinct} distinct value(s). Multi-class labelers need an explicit "
                f"positive-class mapping via the `positive_classes` class attribute."
            )
        result = (
            computed.select([target.pair_col, target.ts_col])
            .with_columns(numeric.alias(LABEL_COL))
            .rename({target.pair_col: "pair", target.ts_col: "ts"})
            .select(["pair", "ts", LABEL_COL])
        )
        return self._restrict(result, at)

    @staticmethod
    def _label_to_numeric(col: pl.Series, positive_classes: tuple[str, ...] = ()) -> pl.Series:
        """Coerce a hard-label series into a numeric event/no-event label."""
        dtype = col.dtype
        if dtype.is_numeric():
            return col.cast(pl.Float64)
        if dtype == pl.Boolean:
            return col.cast(pl.Int64).cast(pl.Float64)

        norm = col.cast(pl.Utf8).str.to_lowercase().str.strip_chars()

        if positive_classes:
            positives = [str(c).lower().strip() for c in positive_classes]
            expr = (
                pl.when(norm.is_null())
                .then(pl.lit(None, dtype=pl.Float64))
                .when(norm.is_in(positives))
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
            )
        else:
            expr = (
                pl.when(norm.is_null())
                .then(pl.lit(None, dtype=pl.Float64))
                .when(norm.is_in(list(_NEGATIVE_LABELS)))
                .then(pl.lit(0.0))
                .otherwise(pl.lit(1.0))
            )
        return pl.select(expr.alias("label")).get_column("label")

    def compute(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute hard labels for an input DataFrame."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute expects pl.DataFrame, got {type(df)}")
        return self._compute_pl(df=df, signals=signals, data_context=data_context)

    def compute_soft(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute soft labels - probability distribution over ``soft_classes``."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute_soft expects pl.DataFrame, got {type(df)}")
        if not self.soft_classes:
            raise NotImplementedError(
                f"{self.__class__.__name__}.soft_classes is empty - declare class names to enable soft labeling"
            )
        return self._compute_soft_pl(df=df, signals=signals, data_context=data_context)

    def _compute_pl(
        self,
        df: pl.DataFrame,
        signals: Signals | None,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        self._validate_input_pl(df)
        df0 = df.sort([self.pair_col, self.ts_col])

        if signals is not None and self.filter_signal_type is not None:
            s_pl = self._signals_to_pl(signals)
            df0 = self._filter_by_signals_pl(df0, s_pl, self.filter_signal_type)

        input_cols = set(df0.columns)

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            out = self.compute_group(g, data_context=data_context)
            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_group must return pl.DataFrame")
            if out.height != g.height:
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
                )
            return out

        out = df0.group_by(self.pair_col, maintain_order=True).map_groups(_wrapped).sort([self.pair_col, self.ts_col])

        if self.keep_input_columns:
            return out

        label_cols = sorted(set(out.columns) - input_cols) if self.output_columns is None else list(self.output_columns)

        keep_cols = [self.pair_col, self.ts_col, *label_cols]
        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def _compute_soft_pl(
        self,
        df: pl.DataFrame,
        signals: Signals | None,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        self._validate_input_pl(df)
        df0 = df.sort([self.pair_col, self.ts_col])

        if signals is not None and self.filter_signal_type is not None:
            s_pl = self._signals_to_pl(signals)
            df0 = self._filter_by_signals_pl(df0, s_pl, self.filter_signal_type)

        soft_cols = [f"{self.soft_col_prefix}{c}" for c in self.soft_classes]

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            out = self.compute_group_soft(g, data_context=data_context)
            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_group_soft must return pl.DataFrame")
            if out.height != g.height:
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
                )
            missing = [c for c in soft_cols if c not in out.columns]
            if missing:
                raise ValueError(f"{self.__class__.__name__}.compute_group_soft missing required columns: {missing}")
            return out

        out = df0.group_by(self.pair_col, maintain_order=True).map_groups(_wrapped).sort([self.pair_col, self.ts_col])

        if self.keep_input_columns:
            return out

        keep_cols = [self.pair_col, self.ts_col, *soft_cols]
        return out.select(keep_cols)

    def _signals_to_pl(self, signals: Signals) -> pl.DataFrame:
        """Convert a signals container/frame to a Polars DataFrame."""
        s = getattr(signals, "value", signals)
        if isinstance(s, pl.DataFrame):
            return s
        raise TypeError(f"Unsupported signals value type: {type(s)}")

    def _filter_by_signals_pl(self, df: pl.DataFrame, s: pl.DataFrame, signal_type: SignalType) -> pl.DataFrame:
        """Filter input to rows matching signal timestamps of a specific type."""
        required = {self.pair_col, self.ts_col, "signal_type"}
        missing = required - set(s.columns)
        if missing:
            raise ValueError(f"Signals missing columns: {sorted(missing)}")

        s_f = (
            s.filter(pl.col("signal_type") == signal_type.value)
            .select([self.pair_col, self.ts_col])
            .unique(subset=[self.pair_col, self.ts_col])
        )
        return df.join(s_f, on=[self.pair_col, self.ts_col], how="inner")

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Soft labels for a single pair - default one-hot of hard labels."""
        hard = self.compute_group(group_df, data_context=data_context)
        if self.out_col not in hard.columns:
            raise ValueError(
                f"{self.__class__.__name__}: default compute_group_soft expects "
                f"out_col={self.out_col!r} in compute_group output"
            )
        label_col = pl.col(self.out_col).cast(pl.Utf8)
        null_label = label_col.is_null()
        soft_exprs = [
            pl.when(null_label)
            .then(pl.lit(None, dtype=pl.Float64))
            .when(label_col == pl.lit(str(cls)))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias(f"{self.soft_col_prefix}{cls}")
            for cls in self.soft_classes
        ]
        return hard.with_columns(soft_exprs)

    @abstractmethod
    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """Compute labels for a single pair group (length-preserving)."""
        raise NotImplementedError

    def _validate_input_pl(self, df: pl.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _apply_signal_mask(
        self,
        df: pl.DataFrame,
        data_context: dict[str, Any],
        group_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Mask labels to signal timestamps only (meta-labeling semantics)."""
        signal_keys: pl.DataFrame = data_context["signal_keys"]
        pair_value = group_df.get_column(self.pair_col)[0]

        signal_ts = signal_keys.filter(pl.col(self.pair_col) == pair_value).select(self.ts_col).unique()

        if signal_ts.height == 0:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(self.out_col))
            if self.include_meta:
                df = df.with_columns([pl.lit(None).alias(col) for col in self.meta_columns])
        else:
            is_signal = pl.col("_is_signal").fill_null(False)
            mask_exprs = [
                pl.when(is_signal)
                .then(pl.col(self.out_col))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias(self.out_col),
            ]
            if self.include_meta:
                mask_exprs += [
                    pl.when(is_signal).then(pl.col(col)).otherwise(pl.lit(None)).alias(col) for col in self.meta_columns
                ]

            df = (
                df.join(
                    signal_ts.with_columns(pl.lit(True).alias("_is_signal")),
                    on=self.ts_col,
                    how="left",
                )
                .with_columns(mask_exprs)
                .drop("_is_signal")
            )

        return df
