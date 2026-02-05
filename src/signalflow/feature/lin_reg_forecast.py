from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Literal

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="forecast/linreg", override=True)
class LinRegForecastFeature(Feature):
    """Enhanced linear regression forecast with trend and mean-reversion features.

    Instead of predicting raw values, predicts change (diff) and adds:
    - Trend slope (recent momentum)
    - Mean reversion signal (deviation from rolling mean)
    - Volatility scaling

    Args:
        source_col: Column to forecast.
        n_lags: Number of lagged diffs. Default: 10.
        trend_window: Window for trend calculation. Default: 5.
        mean_window: Window for mean reversion. Default: 20.
        refit_period: When to refit. Default: 'day'.
        alpha: Ridge regularization. Default: 1.0.
        forecast_horizon: Steps ahead to forecast. Default: 1.
    """

    source_col: str = "rsi_14"
    n_lags: int = 10
    trend_window: int = 5
    mean_window: int = 20
    refit_period: Literal["hour", "day", "week", "month", None] = "day"
    alpha: float = 1.0
    forecast_horizon: int = 1
    min_samples: int = 50

    requires = ["{source_col}"]
    outputs = ["{source_col}_forecast", "{source_col}_forecast_change", "{source_col}_forecast_direction"]

    test_params: ClassVar[list[dict]] = [
        {"source_col": "rsi_14", "n_lags": 10},
        {"source_col": "rsi_14", "n_lags": 5, "mean_window": 10},
    ]

    def __post_init__(self):
        if self.n_lags < 1:
            raise ValueError(f"n_lags must be >= 1")

    def _get_period_key(self, ts: datetime) -> tuple | None:
        if self.refit_period == "hour":
            return (ts.year, ts.month, ts.day, ts.hour)
        elif self.refit_period == "day":
            return (ts.year, ts.month, ts.day)
        elif self.refit_period == "week":
            return (ts.year, ts.isocalendar()[1])
        elif self.refit_period == "month":
            return (ts.year, ts.month)
        return None

    def _build_features(self, values: np.ndarray) -> np.ndarray:
        """Build enhanced feature matrix."""
        n = len(values)

        diffs = np.diff(values, prepend=values[0])

        n_features = self.n_lags + 3
        X = np.full((n, n_features), np.nan, dtype=np.float64)

        start_idx = max(self.n_lags, self.mean_window)

        for i in range(start_idx, n):
            for lag in range(self.n_lags):
                X[i, lag] = diffs[i - lag - 1]

            window = values[i - self.trend_window : i]
            if len(window) == self.trend_window:
                x_trend = np.arange(self.trend_window)
                X[i, self.n_lags] = np.polyfit(x_trend, window, 1)[0]

            mean_window = values[i - self.mean_window : i]
            if len(mean_window) == self.mean_window:
                mean_val = np.mean(mean_window)
                std_val = np.std(mean_window)
                if std_val > 1e-8:
                    X[i, self.n_lags + 1] = (values[i] - mean_val) / std_val
                else:
                    X[i, self.n_lags + 1] = 0

            vol_window = diffs[i - self.trend_window : i]
            if len(vol_window) == self.trend_window:
                X[i, self.n_lags + 2] = np.std(vol_window)

        return X

    def _build_targets(self, values: np.ndarray) -> np.ndarray:
        """Build target: forward diff (change)."""
        n = len(values)
        y = np.full(n, np.nan, dtype=np.float64)

        if self.forecast_horizon < n:
            y[: -self.forecast_horizon] = np.diff(
                values, n=self.forecast_horizon, append=[np.nan] * self.forecast_horizon
            )[: -self.forecast_horizon]
            for i in range(n - self.forecast_horizon):
                y[i] = values[i + self.forecast_horizon] - values[i]

        return y

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute forecasts for single pair."""
        values = df[self.source_col].to_numpy().astype(np.float64)
        timestamps = df[self.ts_col].to_list()
        n = len(values)

        X = self._build_features(values)
        y = self._build_targets(values)

        forecasts = np.full(n, np.nan, dtype=np.float64)
        forecast_changes = np.full(n, np.nan, dtype=np.float64)

        valid_mask = ~np.any(np.isnan(X), axis=1)
        target_valid = ~np.isnan(y)
        train_valid = valid_mask & target_valid

        current_period = None
        model = Ridge(alpha=self.alpha)
        fitted = False

        for i in range(self.min_samples, n):
            if not valid_mask[i]:
                continue

            period = self._get_period_key(timestamps[i])

            if period != current_period:
                current_period = period
                train_idx = np.where(train_valid[:i])[0]
                if len(train_idx) >= 20:
                    model.fit(X[train_idx], y[train_idx])
                    fitted = True

            if fitted:
                predicted_change = model.predict(X[i : i + 1])[0]
                forecast_changes[i] = predicted_change
                forecasts[i] = values[i] + predicted_change

        forecast_col = f"{self.source_col}_forecast"
        change_col = f"{self.source_col}_forecast_change"
        direction_col = f"{self.source_col}_forecast_direction"

        return df.with_columns(
            [
                pl.Series(name=forecast_col, values=forecasts),
                pl.Series(name=change_col, values=forecast_changes),
                pl.Series(name=direction_col, values=np.sign(forecast_changes)),
            ]
        )

    @property
    def warmup(self) -> int:
        return self.min_samples + max(self.n_lags, self.mean_window)
