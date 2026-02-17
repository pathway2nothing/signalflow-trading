from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.core import RawDataType, SfComponentType, SignalCategory, Signals, SignalType


@dataclass
class Labeler(ABC):
    """Base class for Polars-only signal labeling.

    Assigns forward-looking labels to historical data based on future price
    movement. Labels are computed per-pair with length-preserving operations.

    Key concepts:
        - Forward-looking: Labels depend on future data (not available in live trading)
        - Per-pair processing: Each pair labeled independently
        - Length-preserving: Output has same row count as input
        - Signal masking: Optionally label only at signal timestamps

    Public API:
        - compute(): Main entry point (handles grouping, filtering, projection)
        - compute_group(): Per-pair labeling logic (must implement)

    Common labeling strategies:
        - Fixed horizon: Label based on return over N bars
        - Triple barrier: Label based on first hit of profit/loss/time barrier
        - Quantile-based: Label based on return quantiles

    Attributes:
        component_type (ClassVar[SfComponentType]): Always LABELER for registry.
        raw_data_type (RawDataType): Type of raw data. Default: SPOT.
        pair_col (str): Trading pair column. Default: "pair".
        ts_col (str): Timestamp column. Default: "timestamp".
        keep_input_columns (bool): Keep all input columns. Default: False.
        output_columns (list[str] | None): Specific columns to output. Default: None.
        filter_signal_type (SignalType | None): Filter to specific signal type. Default: None.
        mask_to_signals (bool): Mask labels to signal timestamps only. Default: True.
        out_col (str): Output label column name. Default: "label".
        include_meta (bool): Include metadata columns. Default: False.
        meta_columns (tuple[str, ...]): Metadata column names. Default: ("t_hit", "ret").

    Example:
        ```python
        from signalflow.target import Labeler
        from signalflow.core import SignalType
        import polars as pl

        class FixedHorizonLabeler(Labeler):
            '''Label based on fixed-horizon return'''

            def __init__(self, horizon: int = 10, threshold: float = 0.01):
                super().__init__()
                self.horizon = horizon
                self.threshold = threshold

            def compute_group(self, group_df, data_context=None):
                # Compute forward return
                labels = group_df.with_columns([
                    pl.col("close").shift(-self.horizon).alias("future_close")
                ]).with_columns([
                    ((pl.col("future_close") / pl.col("close")) - 1).alias("return")
                ]).with_columns([
                    pl.when(pl.col("return") > self.threshold)
                    .then(pl.lit(SignalType.RISE.value))
                    .when(pl.col("return") < -self.threshold)
                    .then(pl.lit(SignalType.FALL.value))
                    .otherwise(pl.lit(SignalType.NONE.value))
                    .alias("label")
                ])

                return labels

        # Usage
        labeler = FixedHorizonLabeler(horizon=10, threshold=0.01)
        labeled = labeler.compute(ohlcv_df, signals=signals)
        ```

    Note:
        compute_group() must preserve row count (no filtering).
        All timestamps must be timezone-naive.
        Signal masking requires mask_to_signals=True and signal_keys in context.

    See Also:
        FixedHorizonLabeler: Simple fixed-horizon implementation.
        TripleBarrierLabeler: Three-barrier labeling strategy.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.LABELER
    raw_data_type: RawDataType | str = RawDataType.SPOT

    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION
    """Signal category this labeler produces. Default: PRICE_DIRECTION."""

    pair_col: str = "pair"
    ts_col: str = "timestamp"

    keep_input_columns: bool = False
    output_columns: list[str] | None = None
    filter_signal_type: SignalType | None = None

    mask_to_signals: bool = True
    out_col: str = "label"
    include_meta: bool = False
    meta_columns: tuple[str, ...] = ("t_hit", "ret")

    def compute(
        self,
        df: pl.DataFrame,
        signals: Signals | None = None,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute labels for input DataFrame.

        Main entry point - handles validation, filtering, grouping, and projection.

        Processing steps:
            1. Validate input schema
            2. Sort by (pair, timestamp)
            3. (optional) Filter to specific signal type
            4. Group by pair and apply compute_group()
            5. Validate output (length-preserving)
            6. Project to output columns

        Args:
            df (pl.DataFrame): Input data with OHLCV and required columns.
            signals (Signals | None): Signals for filtering/masking.
            data_context (dict[str, Any] | None): Additional context.

        Returns:
            pl.DataFrame: Labeled data with columns:
                - pair, timestamp (always included)
                - label column(s) (as specified by out_col)
                - (optional) metadata columns

        Raises:
            TypeError: If df not pl.DataFrame or compute_group returns wrong type.
            ValueError: If compute_group changes row count or columns missing.

        Example:
            ```python
            # Basic labeling
            labeled = labeler.compute(ohlcv_df)

            # With signal filtering
            labeled = labeler.compute(
                ohlcv_df,
                signals=signals,
                filter_signal_type=SignalType.RISE
            )

            # With masking context
            labeled = labeler.compute(
                ohlcv_df,
                signals=signals,
                data_context={"signal_keys": signal_timestamps_df}
            )
            ```
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute expects pl.DataFrame, got {type(df)}")
        return self._compute_pl(df=df, signals=signals, data_context=data_context)

    def _compute_pl(
        self,
        df: pl.DataFrame,
        signals: Signals | None,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        """Internal Polars-based computation.

        Orchestrates validation, filtering, grouping, and projection.

        Args:
            df (pl.DataFrame): Input data.
            signals (Signals | None): Optional signals.
            data_context (dict[str, Any] | None): Optional context.

        Returns:
            pl.DataFrame: Labeled data.
        """
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

    def _signals_to_pl(self, signals: Signals) -> pl.DataFrame:
        """Convert Signals to Polars DataFrame.

        Args:
            signals (Signals): Signals container.

        Returns:
            pl.DataFrame: Signals as DataFrame.

        Raises:
            TypeError: If Signals.value is not pl.DataFrame.
        """
        s = signals.value
        if isinstance(s, pl.DataFrame):
            return s
        raise TypeError(f"Unsupported Signals.value type: {type(s)}")

    def _filter_by_signals_pl(self, df: pl.DataFrame, s: pl.DataFrame, signal_type: SignalType) -> pl.DataFrame:
        """Filter input to rows matching signal timestamps.

        Inner join with signal timestamps of specific type.

        Args:
            df (pl.DataFrame): Input data.
            s (pl.DataFrame): Signals DataFrame.
            signal_type (SignalType): Signal type to filter.

        Returns:
            pl.DataFrame: Filtered data (only rows at signal timestamps).

        Raises:
            ValueError: If signals missing required columns.
        """
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

    @abstractmethod
    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """Compute labels for single pair group.

        Core labeling logic - must be implemented by subclasses.

        CRITICAL: Must preserve row count (len(output) == len(input)).
        No filtering allowed inside compute_group.

        Args:
            group_df (pl.DataFrame): Single pair's data, sorted by timestamp.
            data_context (dict[str, Any] | None): Additional context.

        Returns:
            pl.DataFrame: Same length as input with added label columns.

        Example:
            ```python
            def compute_group(self, group_df, data_context=None):
                # Compute 10-bar forward return
                return group_df.with_columns([
                    pl.col("close").shift(-10).alias("future_close")
                ]).with_columns([
                    ((pl.col("future_close") / pl.col("close")) - 1).alias("return"),
                    pl.when((pl.col("future_close") / pl.col("close") - 1) > 0.01)
                    .then(pl.lit(SignalType.RISE.value))
                    .otherwise(pl.lit(SignalType.NONE.value))
                    .alias("label")
                ])
            ```

        Note:
            Output must have same height as input (length-preserving).
            Use shift(-n) for forward-looking operations.
            Last N bars will have null labels (no future data).
        """
        raise NotImplementedError

    def _validate_input_pl(self, df: pl.DataFrame) -> None:
        """Validate input DataFrame schema.

        Args:
            df (pl.DataFrame): Input to validate.

        Raises:
            ValueError: If required columns missing.
        """
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _apply_signal_mask(
        self,
        df: pl.DataFrame,
        data_context: dict[str, Any],
        group_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Mask labels to signal timestamps only.

        Labels are computed for all rows, but only signal timestamps
        get actual labels; others are set to SignalType.NONE.

        Used for meta-labeling: only label at detected signal points,
        not every bar.

        Args:
            df (pl.DataFrame): DataFrame with computed labels.
            data_context (dict[str, Any]): Must contain "signal_keys" DataFrame.
            group_df (pl.DataFrame): Original group data for extracting pair value.

        Returns:
            pl.DataFrame: DataFrame with masked labels.

        Example:
            ```python
            # In compute_group with masking
            def compute_group(self, group_df, data_context=None):
                # Compute labels for all rows
                labeled = group_df.with_columns([...])

                # Mask to signal timestamps only
                if self.mask_to_signals and data_context:
                    labeled = self._apply_signal_mask(
                        labeled, data_context, group_df
                    )

                return labeled
            ```

        Note:
            Requires signal_keys in data_context with (pair, timestamp) columns.
            Non-signal rows get label=SignalType.NONE.
            Metadata columns also masked if include_meta=True.
        """
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
