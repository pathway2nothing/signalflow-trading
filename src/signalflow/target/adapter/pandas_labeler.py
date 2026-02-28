from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from signalflow.target.base import Labeler


@dataclass
class PandasLabeler(Labeler):
    """
    Pandas-based labeling implementation, but with the SAME public interface as Labeler:
      extract(pl.DataFrame) -> pl.DataFrame

    Rules:
      - all business logic is implemented on pandas in compute_pd_group()
      - framework stays polars-first externally
      - conversion happens:
          pl group -> pandas group -> pandas out -> polars out
    """

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        pd_in = group_df.to_pandas()
        pd_out = self.compute_pd_group(pd_in, data_context=data_context)

        if not isinstance(pd_out, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__}.compute_pd_group must return pd.DataFrame")
        if len(pd_out) != group_df.height:
            raise ValueError(
                f"{self.__class__.__name__}: len(output_group)={len(pd_out)} != len(input_group)={group_df.height}"
            )

        # IMPORTANT: ensure column order + row order preserved by user implementation
        return pl.from_pandas(pd_out, include_index=False)

    @abstractmethod
    def compute_pd_group(self, group_df: pd.DataFrame, data_context: dict[str, Any] | None) -> pd.DataFrame:
        """
        Pandas labeling per pair.

        MUST:
          - preserve row order
          - preserve row count (no filtering)
        """
        raise NotImplementedError
