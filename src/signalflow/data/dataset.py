"""Dataset - one lazy Polars-backed market-data container."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import polars as pl

from signalflow._logging import frame_summary, step
from signalflow.data.source.base import CANONICAL_COLUMNS, Source
from signalflow.enums import Provenance


class Bar(NamedTuple):
    """One timestamp's cross-section, fed to the decision loop.

    ``prices``/``high``/``low`` map pair -> close/high/low of that bar; ``frame`` is
    the zero-copy slice of the dataset's rows at ``ts``.
    """

    ts: object
    frame: pl.DataFrame
    prices: dict[str, float]
    high: "dict[str, float] | None" = None
    low: "dict[str, float] | None" = None
    open: "dict[str, float] | None" = None


@dataclass(frozen=True)
class Dataset:
    """Immutable view over canonical OHLCV plus any computed columns."""

    frame: pl.DataFrame
    source_name: str = ""
    source_params: dict = field(default_factory=dict)
    quote: str = "USDT"
    provenance: Provenance = Provenance.FULL
    col_provenance: dict = field(default_factory=dict)

    @classmethod
    def from_source(
        cls,
        src: Source,
        pairs: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1h",
        quote: str = "USDT",
    ) -> "Dataset":
        source_name = getattr(src, "name", type(src).__name__)
        with step("Dataset.from_source", source=source_name, interval=interval, pairs=len(pairs)) as log:
            frame = src.fetch(pairs, start, end, interval)
            log["data"] = frame_summary(frame)
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
        """Attach forecast columns keyed by (pair, ts) and record each one's provenance.

        When ``cols`` carries exactly this frame's (pair, ts) index in the same
        order - the shape every ``predict`` emits - the columns are appended
        without a join, so the raw frame is neither copied nor reordered.
        Otherwise a left join aligns them.
        """
        new_cols = [c for c in cols.columns if c not in ("pair", "ts")]
        if _same_index(self.frame, cols):
            merged = self.frame.with_columns([cols.get_column(c) for c in new_cols])
        else:
            merged = self.frame.join(cols, on=["pair", "ts"], how="left")
        col_prov = {**self.col_provenance, **{c: Provenance(provenance).value for c in new_cols}}
        return replace(self, frame=merged, provenance=provenance, col_provenance=col_prov)

    def with_oos_forecasts(self, model) -> "Dataset":
        """Attach a model's leak-free out-of-fold predictions, stamped OOS by construction."""
        pred = model.predict_oos(self)
        return self.with_forecasts(pred, provenance=Provenance.OOS)

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
        """Yield one :class:`Bar` per timestamp in order (the replay backbone).

        The frame is stably sorted by ``ts`` once (pairs keep their order inside a
        timestamp); every bar is then a zero-copy ``slice`` at a precomputed offset,
        so the per-bar cost is independent of the dataset size.
        """
        frame = self.frame
        if columns is not None:
            keep = ["pair", "ts", "close", *[c for c in ("high", "low", "open") if c in frame.columns], *columns]
            frame = frame.select(list(dict.fromkeys(keep)))
        if frame.height == 0:
            return
        frame = frame.sort("ts", maintain_order=True)
        groups = frame.group_by("ts", maintain_order=True).agg(pl.len().alias("n"))
        ts_values = groups.get_column("ts").to_list()
        lengths = groups.get_column("n").to_list()
        pairs = frame.get_column("pair").to_list()
        close = frame.get_column("close").to_list()
        high = frame.get_column("high").to_list() if "high" in frame.columns else None
        low = frame.get_column("low").to_list() if "low" in frame.columns else None
        open_ = frame.get_column("open").to_list() if "open" in frame.columns else None
        offset = 0
        for ts_val, n in zip(ts_values, lengths, strict=True):
            end = offset + n
            keys = pairs[offset:end]
            yield Bar(
                ts=ts_val,
                frame=frame.slice(offset, n),
                prices=dict(zip(keys, close[offset:end], strict=True)),
                high=dict(zip(keys, high[offset:end], strict=True)) if high is not None else None,
                low=dict(zip(keys, low[offset:end], strict=True)) if low is not None else None,
                open=dict(zip(keys, open_[offset:end], strict=True)) if open_ is not None else None,
            )
            offset = end


def _same_index(frame: pl.DataFrame, cols: pl.DataFrame) -> bool:
    """True when both frames hold identical (pair, ts) columns row for row."""
    if cols.height != frame.height or "pair" not in cols.columns or "ts" not in cols.columns:
        return False
    return frame.get_column("pair").equals(cols.get_column("pair")) and frame.get_column("ts").equals(
        cols.get_column("ts")
    )


_FLOAT_DTYPES = {"f32": pl.Float32, "float32": pl.Float32, "f64": pl.Float64, "float64": pl.Float64}


def data(
    source: str | Source,
    pairs: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1h",
    quote: str = "USDT",
    cache_dir: "str | Path | None" = None,
    cache_partition: "str | None" = None,
    dtype: "str | type[pl.DataType] | None" = None,
    **source_kwargs,
) -> Dataset:
    """Build a Dataset from a registered source name or a Source instance.

    When ``cache_dir`` is set the resolved source is wrapped in a disk cache that
    fetches only spans absent from ``<cache_dir>/<interval>/<PAIR>.parquet``.
    ``cache_partition="day"`` switches that cache to one file per (pair, day) -
    ``<cache_dir>/<interval>/<PAIR>/<YYYY-MM-DD>.parquet`` - so every completed
    day persists immediately and an interrupted fetch keeps its progress.
    ``dtype="f32"`` downcasts every float column after load (half the memory).
    """
    from signalflow.enums import ComponentType
    from signalflow.registry import registry

    src = registry.create(ComponentType.SOURCE, source, **source_kwargs) if isinstance(source, str) else source
    if cache_dir is not None:
        from signalflow.data.source.cached import CachedSource

        src = CachedSource(src, cache_dir, partition=cache_partition)
    elif cache_partition is not None:
        raise ValueError("cache_partition requires cache_dir")
    ds = Dataset.from_source(src, pairs=pairs, start=start, end=end, interval=interval, quote=quote)
    if dtype is not None:
        target = _FLOAT_DTYPES.get(dtype.lower()) if isinstance(dtype, str) else dtype
        if target is None:
            raise ValueError(f"unsupported dtype {dtype!r}; use 'f32'/'f64'")
        ds = ds.with_frame(ds.frame.with_columns(pl.col(pl.Float64).cast(target)))
    return ds


__all__ = ["CANONICAL_COLUMNS", "Bar", "Dataset", "data"]
