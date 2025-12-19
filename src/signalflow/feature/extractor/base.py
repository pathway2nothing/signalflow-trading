# from __future__ import annotations

# from abc import ABC, abstractmethod
# from dataclasses import dataclass
# from typing import Any, Literal

# import polars as pl
# import pandas as pd

# from signalflow.core.offset_resampler import OffsetResampler


# @dataclass
# class FeatureExtractor(ABC):
#     """
#     Base FeatureExtractor.

#     Responsibilities:
#       - validate required columns
#       - ensure `resample_offset` exists (via OffsetResampler.add_offset_column)
#       - optionally run sliding offset-resample (via OffsetResampler.resample)
#       - optionally keep only last offset (production)
#       - group by (pair, resample_offset) and map compute_group()
#       - strict invariant: per-group output length == input length
#       - final sort by (pair, timestamp)
#     """

#     window_minutes: int
#     compute_last_offset: bool = False

#     pair_col: str = "pair"
#     ts_col: str = "timestamp"
#     offset_col: str = "resample_offset"

#     # Optional preprocessing (sliding resample)
#     use_resample: bool = False
#     resample_mode: Literal["add", "replace"] = "add"
#     resample_prefix: str | None = None
#     data_type: str = "spot"

#     def __post_init__(self) -> None:
#         if self.window_minutes <= 0:
#             raise ValueError(f"window_minutes must be > 0, got {self.window_minutes}")

#         if self.resample_mode not in ("add", "replace"):
#             raise ValueError(f"Invalid resample_mode: {self.resample_mode}")

#         if self.offset_col != OffsetResampler.OFFSET_COL:
#             raise ValueError(
#                 f"offset_col must be '{OffsetResampler.OFFSET_COL}', got '{self.offset_col}'"
#             )

#     @property
#     def _resampler(self) -> OffsetResampler:
#         return OffsetResampler(
#             window_minutes=self.window_minutes,
#             ts_col=self.ts_col,
#             pair_col=self.pair_col,
#             mode=self.resample_mode,
#             prefix=self.resample_prefix,
#             data_type=self.data_type,
#         )

#     def extract(self, df: pl.DataFrame | pd.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame | pd.DataFrame:
#         if isinstance(df, pl.DataFrame):
#             return self._extract_pl(df, data_context=data_context)
#         if isinstance(df, pd.DataFrame):
#             return self._extract_pd(df, data_context=data_context)
#         raise TypeError(f"Unsupported df type: {type(df)}")

#     def _extract_pl(self, df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
#         self._validate_input_pl(df)

#         if not isinstance(self.pair_col, str):
#             raise TypeError(f"pair_col must be str, got {type(self.pair_col)}: {self.pair_col}")
#         if not isinstance(self.offset_col, str):
#             raise TypeError(f"offset_col must be str, got {type(self.offset_col)}: {self.offset_col}")
#         if not isinstance(self.ts_col, str):
#             raise TypeError(f"ts_col must be str, got {type(self.ts_col)}: {self.ts_col}")

#         df0 = df.sort([self.pair_col, self.ts_col])

#         if self.offset_col not in df0.columns:
#             df0 = self._resampler.add_offset_column(df0)  

#         if self.use_resample:
#             df0 = self._resampler.resample(df0)  

#         if self.compute_last_offset:
#             last_off = self._resampler.get_last_offset(df0)
#             df0 = df0.filter(pl.col(self.offset_col) == last_off)

#         def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
#             out = self.compute_pl_group(g, data_context=data_context)
#             if out.height != g.height:
#                 raise ValueError(
#                     f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
#                 )
#             return out

#         out = (
#             df0
#             .group_by(self.pair_col, self.offset_col, maintain_order=True)
#             .map_groups(_wrapped)
#         )

#         return out.sort([self.pair_col, self.ts_col])


#     @abstractmethod
#     def compute_pl_group(
#         self,
#         group_df: pl.DataFrame,
#         data_context: dict[str, Any] | None,
#     ) -> pl.DataFrame:
#         """
#         Pure group transform.

#         group_df:
#           - one pair
#           - one resample_offset
#           - already prepared (offset exists; resample exists if use_resample=True)

#         MUST:
#           - preserve row order
#           - len(output) == len(input)
#           - do not filter rows
#         """
#         raise NotImplementedError

#     def _validate_input_pl(self, df: pl.DataFrame) -> None:
#         missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")


#     def _extract_pd(self, df: pd.DataFrame, data_context: dict[str, Any] | None) -> pd.DataFrame:
#         self._validate_input_pd(df)

#         df0 = df.sort_values([self.pair_col, self.ts_col], kind="stable").copy()

#         if self.offset_col not in df0.columns:
#             df0 = self._resampler.add_offset_column(df0)  # type: ignore[assignment]

#         if self.use_resample:
#             df0 = self._resampler.resample(df0)  # type: ignore[assignment]

#         if self.compute_last_offset:
#             last_off = self._resampler.get_last_offset(df0)
#             df0 = df0[df0[self.offset_col] == last_off].copy()

#         parts: list[pd.DataFrame] = []
#         for (_, _), g in df0.groupby([self.pair_col, self.offset_col], sort=False, dropna=False):
#             out_g = self.compute_pd_group(g, data_context=data_context)
#             if not isinstance(out_g, pd.DataFrame):
#                 raise TypeError(f"{self.__class__.__name__}.compute_pd_group must return pd.DataFrame")
#             if len(out_g) != len(g):
#                 raise ValueError(
#                     f"{self.__class__.__name__}: len(output_group)={len(out_g)} != len(input_group)={len(g)}"
#                 )
#             parts.append(out_g)

#         out = pd.concat(parts, axis=0, ignore_index=True) if parts else df0.iloc[:0].copy()

#         return out.sort_values([self.pair_col, self.ts_col], kind="stable").reset_index(drop=True)

#     def compute_pd_group(
#         self,
#         group_df: pd.DataFrame,
#         data_context: dict[str, Any] | None,
#     ) -> pd.DataFrame:
#         """
#         Default pandas path: convert to polars -> compute_pl_group -> back.
#         Override for native pandas speed.
#         """
#         pl_in = pl.from_pandas(group_df)
#         pl_out = self.compute_pl_group(pl_in, data_context=data_context)
#         if pl_out.height != pl_in.height:
#             raise ValueError(
#                 f"{self.__class__.__name__}: len(output_group)={pl_out.height} != len(input_group)={pl_in.height}"
#             )
#         return pl_out.to_pandas()

#     def _validate_input_pd(self, df: pd.DataFrame) -> None:
#         missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl
import pandas as pd

from signalflow.core.offset_resampler import OffsetResampler


DfLike = pl.DataFrame | pd.DataFrame


@dataclass
class FeatureExtractor(ABC):
    """
    Base FeatureExtractor.

    Steps:
      1) sort (pair, timestamp)
      2) ensure resample_offset
      3) optional sliding resample
      4) optional last-offset filter
      5) group (pair, resample_offset) and compute features
      6) sort output
      7) optional projection:
           - keep_input_columns=True  -> return everything
           - keep_input_columns=False -> return only [pair, timestamp] + new feature cols
    """

    offset_window: int
    compute_last_offset: bool = False

    pair_col: str = "pair"
    ts_col: str = "timestamp"
    offset_col: str = "resample_offset"

    use_resample: bool = False
    resample_mode: Literal["add", "replace"] = "add"
    resample_prefix: str | None = None
    data_type: str = "spot"

    keep_input_columns: bool = False  

    def __post_init__(self) -> None:
        if self.offset_window <= 0:
            raise ValueError(f"window_minutes must be > 0, got {self.offset_window}")
        if self.resample_mode not in ("add", "replace"):
            raise ValueError(f"Invalid resample_mode: {self.resample_mode}")
        if self.offset_col != OffsetResampler.OFFSET_COL:
            raise ValueError(f"offset_col must be '{OffsetResampler.OFFSET_COL}', got '{self.offset_col}'")
        if not isinstance(self.pair_col, str) or not isinstance(self.ts_col, str) or not isinstance(self.offset_col, str):
            raise TypeError("pair_col/ts_col/offset_col must be str")

    @property
    def _resampler(self) -> OffsetResampler:
        return OffsetResampler(
            window_minutes=self.offset_window,
            ts_col=self.ts_col,
            pair_col=self.pair_col,
            mode=self.resample_mode,
            prefix=self.resample_prefix,
            data_type=self.data_type,
        )

    def extract(self, df: DfLike, data_context: dict[str, Any] | None = None) -> DfLike:
        if isinstance(df, pl.DataFrame):
            return self._extract_pl(df, data_context=data_context)
        if isinstance(df, pd.DataFrame):
            return self._extract_pd(df, data_context=data_context)
        raise TypeError(f"Unsupported df type: {type(df)}")

    def _extract_pl(self, df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        self._validate_input_pl(df)

        df0 = df.sort([self.pair_col, self.ts_col])

        if self.offset_col not in df0.columns:
            df0 = self._resampler.add_offset_column(df0)  

        if self.use_resample:
            df0 = self._resampler.resample(df0) 

        if self.compute_last_offset:
            last_off = self._resampler.get_last_offset(df0)
            df0 = df0.filter(pl.col(self.offset_col) == last_off)

        prepared_cols = set(df0.columns)
        inferred_features: set[str] = set()

        def _wrapped(g: pl.DataFrame) -> pl.DataFrame:
            nonlocal inferred_features

            in_cols = set(g.columns)
            out = self.compute_pl_group(g, data_context=data_context)

            if not isinstance(out, pl.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_pl_group must return pl.DataFrame")
            if out.height != g.height:
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={out.height} != len(input_group)={g.height}"
                )

            if not inferred_features:
                inferred_features = set(out.columns) - in_cols

            return out

        out = (
            df0
            .group_by(self.pair_col, self.offset_col, maintain_order=True)
            .map_groups(_wrapped)
            .sort([self.pair_col, self.ts_col])
        )

        if self.keep_input_columns:
            return out

        feature_cols = sorted(set(out.columns) - prepared_cols)
        keep_cols = [self.pair_col, self.ts_col] + feature_cols

        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.select(keep_cols)

    def compute_pl_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None,
    ) -> pl.DataFrame:
        raise NotImplementedError

    def _validate_input_pl(self, df: pl.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _extract_pd(self, df: pd.DataFrame, data_context: dict[str, Any] | None) -> pd.DataFrame:
        self._validate_input_pd(df)

        df0 = df.sort_values([self.pair_col, self.ts_col], kind="stable").copy()

        if self.offset_col not in df0.columns:
            df0 = self._resampler.add_offset_column(df0)  

        if self.use_resample:
            df0 = self._resampler.resample(df0)  

        if self.compute_last_offset:
            last_off = self._resampler.get_last_offset(df0)
            df0 = df0[df0[self.offset_col] == last_off].copy()

        prepared_cols = set(df0.columns)

        parts: list[pd.DataFrame] = []
        for (_, _), g in df0.groupby([self.pair_col, self.offset_col], sort=False, dropna=False):
            out_g = self.compute_pd_group(g, data_context=data_context)
            if not isinstance(out_g, pd.DataFrame):
                raise TypeError(f"{self.__class__.__name__}.compute_pd_group must return pd.DataFrame")
            if len(out_g) != len(g):
                raise ValueError(
                    f"{self.__class__.__name__}: len(output_group)={len(out_g)} != len(input_group)={len(g)}"
                )
            parts.append(out_g)

        out = pd.concat(parts, axis=0, ignore_index=True) if parts else df0.iloc[:0].copy()
        out = out.sort_values([self.pair_col, self.ts_col], kind="stable").reset_index(drop=True)

        if self.keep_input_columns:
            return out

        feature_cols = [c for c in out.columns if c not in prepared_cols]
        keep_cols = [self.pair_col, self.ts_col] + feature_cols

        missing = [c for c in keep_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Projection error, missing columns: {missing}")

        return out.loc[:, keep_cols]

    def compute_pd_group(
        self,
        group_df: pd.DataFrame,
        data_context: dict[str, Any] | None,
    ) -> pd.DataFrame:
        pl_in = pl.from_pandas(group_df)
        pl_out = self.compute_pl_group(pl_in, data_context=data_context)
        if pl_out.height != pl_in.height:
            raise ValueError(
                f"{self.__class__.__name__}: len(output_group)={pl_out.height} != len(input_group)={pl_in.height}"
            )
        return pl_out.to_pandas()

    def _validate_input_pd(self, df: pd.DataFrame) -> None:
        missing = [c for c in (self.pair_col, self.ts_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
