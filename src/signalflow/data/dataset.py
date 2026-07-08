"""Dataset - one lazy Polars-backed market-data container."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import polars as pl

from signalflow.data.source.base import CANONICAL_COLUMNS, Source
from signalflow.enums import Provenance


class Bar(NamedTuple):
    """One timestamp's cross-section, fed to the decision loop."""

    ts: object
    frame: pl.DataFrame
    prices: dict[str, float]


@dataclass(frozen=True)
class Dataset:
    """Immutable view over canonical OHLCV plus any computed columns."""

    frame: pl.DataFrame
    source_name: str = ""
    source_params: dict = field(default_factory=dict)
    quote: str = "USDT"
    provenance: Provenance = Provenance.FULL

    @classmethod
    def from_source(
        cls,
        src: Source,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1m",
        quote: str = "USDT",
    ) -> "Dataset":
        frame = src.fetch(pairs, start, end, interval)
        return cls(
            frame=frame,
            source_name=getattr(src, "name", ""),
            source_params={"pairs": pairs, "start": start, "end": end, "interval": interval},
            quote=quote,
        )

    def pairs(self) -> list[str]:
        return self.frame.get_column("pair").unique(maintain_order=True).to_list()

    def index(self) -> pl.DataFrame:
        """(pair, ts) of every row - the universe a sampler selects from."""
        return self.frame.select(["pair", "ts"])

    def lazy(self) -> pl.LazyFrame:
        return self.frame.lazy()

    def columns(self) -> list[str]:
        return self.frame.columns

    @property
    def height(self) -> int:
        return self.frame.height

    def with_frame(self, frame: pl.DataFrame, *, provenance: Provenance | None = None) -> "Dataset":
        return replace(self, frame=frame, provenance=provenance or self.provenance)

    def with_forecasts(self, cols: pl.DataFrame, *, provenance: Provenance = Provenance.FULL) -> "Dataset":
        """Join forecast columns (keyed by pair, ts) and record their provenance."""
        merged = self.frame.join(cols, on=["pair", "ts"], how="left")
        return replace(self, frame=merged, provenance=provenance)

    def slice_time(self, start=None, end=None) -> "Dataset":
        """Rows with start <= ts < end (either bound optional). For walk-forward windows."""
        expr = pl.lit(True)
        if start is not None:
            expr = expr & (pl.col("ts") >= start)
        if end is not None:
            expr = expr & (pl.col("ts") < end)
        return replace(self, frame=self.frame.filter(expr))

    def select_pairs(self, pairs: list[str]) -> "Dataset":
        return replace(self, frame=self.frame.filter(pl.col("pair").is_in(pairs)))

    def prices_at(self, ts) -> dict[str, float]:
        slice_ = self.frame.filter(pl.col("ts") == ts)
        return dict(zip(slice_.get_column("pair"), slice_.get_column("close"), strict=True))

    def cross_rate(self, base: str, quote: str, prices: dict[str, float]) -> float:
        """Price of ``base`` denominated in ``quote`` given a pair->close map."""
        from signalflow.engine.types import cross_rate

        return cross_rate(base, quote, prices)

    def iter_bars(self, columns: list[str] | None = None) -> Iterator[Bar]:
        """Yield one :class:`Bar` per timestamp in order (the replay backbone)."""
        frame = self.frame if columns is None else self.frame.select(["pair", "ts", "close", *columns])
        for ts, sub in frame.sort("ts").group_by("ts", maintain_order=True):
            ts_val = ts[0] if isinstance(ts, tuple) else ts
            prices = dict(zip(sub.get_column("pair"), sub.get_column("close"), strict=True))
            yield Bar(ts=ts_val, frame=sub, prices=prices)


def data(
    source: str | Source,
    pairs: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1m",
    quote: str = "USDT",
    cache_dir: "str | Path | None" = None,
    **source_kwargs,
) -> Dataset:
    """Build a Dataset from a registered source name or a Source instance.

    When ``cache_dir`` is set the resolved source is wrapped in a disk cache that
    fetches only spans absent from ``<cache_dir>/<interval>/<PAIR>.parquet``.
    """
    from signalflow.enums import ComponentType
    from signalflow.registry import registry

    src = registry.create(ComponentType.SOURCE, source, **source_kwargs) if isinstance(source, str) else source
    if cache_dir is not None:
        from signalflow.data.source.cached import CachedSource

        src = CachedSource(src, cache_dir)
    return Dataset.from_source(src, pairs=pairs, start=start, end=end, interval=interval, quote=quote)


__all__ = ["CANONICAL_COLUMNS", "Bar", "Dataset", "data"]
