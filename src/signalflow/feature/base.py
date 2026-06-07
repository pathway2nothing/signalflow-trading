from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from signalflow.core import SfComponentType
from signalflow.utils import KwargsTolerantMixin

if TYPE_CHECKING:
    from signalflow.core import RawData


@dataclass
class Feature(KwargsTolerantMixin):
    """Base class for all features.

    Two methods to implement:
        - compute(df): all pairs, abstract for GlobalFeature/Pipeline
        - compute_pair(df): one pair, for regular features

    Attributes:
        requires: Input column templates, e.g. ["{price_col}"]
        outputs: Output column templates, e.g. ["rsi_{period}"]
        normalized: If True, apply rolling z-score normalization to output.
        norm_period: Window for normalization. Defaults to 3x feature period.
    """

    component_type: ClassVar[SfComponentType] = SfComponentType.FEATURE
    requires: ClassVar[list[str]] = []
    outputs: ClassVar[list[str]] = []
    test_params: ClassVar[list[dict]] = []

    # --- Warmup reproducibility contract (Stage 3) ---
    # is_recursive: indicator is stateful / entry-point dependent (its value at
    #   bar N depends on the start of the series unless correctly initialized).
    # warmup_invariant: indicator guarantees entry-point invariance after warmup
    #   (e.g. via deterministic SMA-seeded initialization). For recursive features
    #   that do NOT re-seed deterministically this must be False — otherwise live
    #   and backtest values diverge and break parity.
    is_recursive: ClassVar[bool] = False
    warmup_invariant: ClassVar[bool] = True

    group_col: str = "pair"
    ts_col: str = "timestamp"
    normalized: bool = False
    norm_period: int | None = None

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute feature for all pairs"""
        return df.group_by(self.group_col, maintain_order=True).map_groups(self.compute_pair)

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute feature for single pair. Override for per-pair features."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute_pair()")

    def output_cols(self, prefix: str = "") -> list[str]:
        """Actual output column names with parameter substitution."""
        return [f"{prefix}{tpl.format(**self.__dict__)}" for tpl in self.outputs]

    def required_cols(self) -> list[str]:
        """Actual required column names with parameter substitution."""
        return [tpl.format(**self.__dict__) if "{" in tpl else tpl for tpl in self.requires]

    @property
    def warmup(self) -> int:
        """Minimum bars needed before output is stable.

        Override in subclasses with feature-specific logic.
        Default: 0 (no warmup required).
        """
        return 0

    def assert_reproducible(self) -> None:
        """Assert this feature honours the warmup reproducibility contract.

        A recursive feature that does not guarantee entry-point invariance
        (``is_recursive and not warmup_invariant``) will produce different
        values in live vs. backtest depending on the warmup start point,
        breaking parity. Such a feature raises here so the problem surfaces
        before it reaches production.

        Raises:
            RuntimeError: if the feature is recursive and not warmup-invariant.
        """
        if self.is_recursive and not self.warmup_invariant:
            raise RuntimeError(
                f"{self.__class__.__name__} is recursive (entry-point dependent) and does NOT "
                f"guarantee warmup invariance (warmup_invariant=False). Its value at a given bar "
                f"depends on where the warmup window starts, so live values will diverge from "
                f"backtest and break parity. Use a deterministically seeded (e.g. SMA-init) "
                f"implementation, or extend warmup until convergence is provably within tolerance."
            )


@dataclass
class GlobalFeature(Feature):
    """Base class for features computed across all pairs.

    Override compute() with custom aggregation logic.

    For multi-source features, set `sources` to specify which exchanges to use.
    Use `get_source_data()` to retrieve data from RawData with proper handling.

    Attributes:
        sources: List of source names to use (e.g., ["binance", "okx"]).
            If None, uses default source or all available sources.

    Example:
        ```python
        @dataclass
        class AggregatedOI(GlobalFeature):
            sources: list[str] | None = None

            def compute_from_raw(self, raw: RawData, context=None) -> pl.DataFrame:
                # Get data from specified sources
                for source, df in self.iter_sources(raw, "perpetual"):
                    ...
        ```
    """

    sources: list[str] | None = field(default=None)

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Must override - compute global feature across all pairs."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")

    def get_source_data(
        self,
        raw: "RawData",
        data_type: str,
        source: str | None = None,
    ) -> pl.DataFrame:
        """Get DataFrame from RawData for a specific source.

        Args:
            raw: RawData container.
            data_type: Data type key (e.g., "perpetual", "spot").
            source: Specific source name. If None, uses default.

        Returns:
            pl.DataFrame: Data for the specified source.
        """
        if source is not None:
            return raw.get(data_type, source=source)
        return raw.get(data_type)

    def iter_sources(
        self,
        raw: "RawData",
        data_type: str,
    ) -> Iterator[tuple[str, pl.DataFrame]]:
        """Iterate over source DataFrames from RawData.

        If `self.sources` is set, iterates only those sources.
        Otherwise, iterates all available sources.

        Args:
            raw: RawData container.
            data_type: Data type key (e.g., "perpetual").

        Yields:
            tuple[str, pl.DataFrame]: (source_name, DataFrame) pairs.

        Example:
            ```python
            for source, df in self.iter_sources(raw, "perpetual"):
                print(f"{source}: {df.shape}")
            ```
        """
        if data_type not in raw:
            return

        accessor = getattr(raw, data_type)

        # Determine which sources to iterate
        sources_to_use = self.sources if self.sources else accessor.sources

        for source in sources_to_use:
            if source in accessor:
                yield source, getattr(accessor, source)
