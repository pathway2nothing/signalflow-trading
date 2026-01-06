# IMPORTANT

from dataclasses import dataclass
import polars as pl
from signalflow.feature.base import FeatureExtractor
from signalflow.core import sf_component


@dataclass
@sf_component(name="rsi")
class RsiExtractor(FeatureExtractor):
    rsi_period: int = 14
    price_col: str = "close"
    out_col: str = "rsi"
    use_resample:bool = True

    def compute_group(self, group_df: pl.DataFrame, data_context: dict | None) -> pl.DataFrame:
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