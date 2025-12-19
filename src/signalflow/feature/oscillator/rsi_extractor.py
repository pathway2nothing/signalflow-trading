from dataclasses import dataclass
import polars as pl
import pandas as pd
from signalflow.feature.base_extractor import FeatureExtractor


@dataclass
class RsiExtractor(FeatureExtractor):
    rsi_period: int = 14
    price_col: str = "close"
    out_col: str = "rsi"
    use_resample:bool = True

    def compute_pl_group(self, group_df: pl.DataFrame, data_context: dict | None) -> pl.DataFrame:
        price = pl.col(self.price_col)
        delta = price.diff()

        gain = delta.clip(lower_bound=0.0)
        loss = (-delta).clip(lower_bound=0.0)

        avg_gain = gain.rolling_mean(
            window_size=self.rsi_period,
            min_samples=self.rsi_period,
        )
        avg_loss = loss.rolling_mean(
            window_size=self.rsi_period,
            min_samples=self.rsi_period,
        )


        rs = avg_gain / avg_loss

        rsi = (
            pl.when((avg_loss == 0) & (avg_gain == 0)).then(50.0)
            .when(avg_loss == 0).then(100.0)
            .otherwise(100.0 - (100.0 / (1.0 + rs)))
        )

        return group_df.with_columns(rsi.alias(self.out_col))

    def compute_pd_group(
        self,
        group_df: pd.DataFrame,
        data_context: dict | None,
    ) -> pd.DataFrame:
        """
        RSI computed per (pair, resample_offset) group.
        Length and order are preserved.
        """

        price = pd.to_numeric(group_df[self.price_col], errors="coerce")

        delta = price.diff()

        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.rolling(
            window=self.rsi_period,
            min_periods=self.rsi_period,
        ).mean()

        avg_loss = loss.rolling(
            window=self.rsi_period,
            min_periods=self.rsi_period,
        ).mean()

        rs = avg_gain / avg_loss

        rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0
        rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0

        out = group_df.copy()
        out[self.out_col] = rsi

        return out