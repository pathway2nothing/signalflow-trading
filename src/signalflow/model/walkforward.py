"""Walk-forward orchestration: calendar-stepped train/evaluate folds with per-fold models.

Replaces the rolling monthly loop every experiment reimplements::

    result = walk_forward(model, data, train="90d", step="30d")
    scores = result.evaluate(my_metric)
    oos = result.oos()
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from loguru import logger

from signalflow.data.dataset import Dataset
from signalflow.experiment.cache import ArtifactCache
from signalflow.model.forecast import ForecastModel
from signalflow.model.oos import parse_duration
from signalflow.target import LABEL_COL

_METRIC_PRESETS = ("auc", "pr_auc", "brier")


def _resolve_metric(name: str, output_col: str) -> "Callable[[pl.DataFrame], float]":
    """Build a per-fold scorer for a preset name (``auc``/``pr_auc``/``brier``)."""
    if name not in _METRIC_PRESETS:
        raise ValueError(f"unknown metric preset {name!r}; choose from {list(_METRIC_PRESETS)} or pass a callable")
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    def score(frame: pl.DataFrame) -> float:
        df = frame.drop_nulls(subset=[output_col, LABEL_COL])
        y = df.get_column(LABEL_COL).cast(pl.Int8).to_numpy()
        p = df.get_column(output_col).to_numpy()
        if name == "brier":
            return float(brier_score_loss(y, p)) if y.size else float("nan")
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p)) if name == "auc" else float(average_precision_score(y, p))

    return score


@dataclass
class WalkForwardFold:
    """One train-on-trailing-window / evaluate-next-window step and its fitted model."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    model: "ForecastModel"
    oos: pl.DataFrame


@dataclass
class WalkForwardResult:
    """Per-fold models and out-of-sample predictions from a walk-forward run."""

    folds: list[WalkForwardFold]

    def oos(self) -> pl.DataFrame:
        """Merged out-of-sample rows across folds, deduped by (pair, ts)."""
        parts = [f.oos for f in self.folds if f.oos.height > 0]
        if not parts:
            return pl.DataFrame()
        return pl.concat(parts).unique(subset=["pair", "ts"], keep="first").sort(["pair", "ts"])

    def evaluate(self, metric: "Callable[[pl.DataFrame], float] | str") -> pl.DataFrame:
        """One row per fold: its window bounds and ``metric(fold.oos)``.

        ``metric`` is a callable on a fold's OOS frame, or a preset name
        (``"auc"``, ``"pr_auc"``, ``"brier"``); an unknown name raises ``ValueError``.
        """
        rows = [
            {
                "fold": i,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "score": float((_resolve_metric(metric, f.model.output) if isinstance(metric, str) else metric)(f.oos)),
            }
            for i, f in enumerate(self.folds)
        ]
        return pl.DataFrame(rows)


def _add_months(moment: datetime, months: int) -> datetime:
    """Calendar-month arithmetic clamping the day into the target month."""
    total = moment.month - 1 + months
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, _days_in_month(year, month))
    return moment.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    nxt = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return (nxt - datetime(year, month, 1)).days


def _advance(moment: datetime, span: str) -> datetime:
    """Step ``moment`` forward by a duration string; ``"1mo"``/``"3mo"`` are calendar months."""
    text = span.strip().lower()
    if text.endswith("mo"):
        return _add_months(moment, int(text[:-2]))
    return moment + parse_duration(text)


def _retreat(moment: datetime, span: str) -> datetime:
    text = span.strip().lower()
    if text.endswith("mo"):
        return _add_months(moment, -int(text[:-2]))
    return moment - parse_duration(text)


def _bounds(data: Dataset, start: "datetime | None", end: "datetime | None") -> "tuple[datetime, datetime]":
    ts = data.frame.get_column("ts")
    first = start if start is not None else ts.min()
    last = end if end is not None else ts.max()
    return first, last


def _windows(
    data: Dataset, train: str, step: str, start: "datetime | None", end: "datetime | None"
) -> "list[tuple[datetime, datetime, datetime, datetime]]":
    first, last = _bounds(data, start, end)
    windows: list[tuple[datetime, datetime, datetime, datetime]] = []
    test_start = _advance(first, train)
    while test_start < last:
        test_end = min(_advance(test_start, step), last)
        train_start = _retreat(test_start, train)
        windows.append((train_start, test_start, test_start, test_end))
        test_start = test_end
    return windows


def _test_slice(data: Dataset, test_start: datetime, test_end: datetime, warmup: int) -> Dataset:
    """Test window plus the trailing ``warmup`` bars per pair so features start hot."""
    frame = data.frame.filter(pl.col("ts") < test_end)
    if warmup <= 0:
        return Dataset(frame=frame.filter(pl.col("ts") >= test_start), quote=data.quote)
    history = frame.filter(pl.col("ts") < test_start)
    lookback = history.filter(pl.col("ts").rank("ordinal", descending=True).over("pair") <= warmup)
    short = lookback.group_by("pair").len().filter(pl.col("len") < warmup)
    if short.height:
        logger.warning(
            f"walk_forward: pairs with less than {warmup} lookback bars before {test_start}: "
            f"{short.get_column('pair').to_list()}"
        )
    window = frame.filter(pl.col("ts") >= test_start)
    return Dataset(frame=pl.concat([lookback, window]).sort(["pair", "ts"]), quote=data.quote)


def walk_forward(
    model: "ForecastModel",
    data: Dataset,
    train: str,
    step: str,
    start: "datetime | None" = None,
    end: "datetime | None" = None,
    save_to: "str | None" = None,
    cache: "ArtifactCache | None" = None,
    feature_store=None,
) -> WalkForwardResult:
    """Calendar-stepped walk-forward: refit ``model`` per fold on the trailing ``train`` window.

    ``model`` is a declarative template (unfitted); each fold clones it, fits on
    ``[test_start - train, test_start)`` and predicts ``[test_start, test_start + step)``.
    ``save_to`` may template the fold index, e.g. ``"mlflow://models/exp_{fold}"``.
    """
    windows = _windows(data, train, step, start, end)
    labels_all = model.target.labels(data) if model.target is not None else None
    folds: list[WalkForwardFold] = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        fold_model = copy.deepcopy(model)
        train_ds = data.slice_time(train_start, train_end)
        fold_model.fit(train_ds, cache=cache, feature_store=feature_store)
        test_ds = _test_slice(data, test_start, test_end, fold_model.features.warmup)
        preds = fold_model.predict(test_ds)
        oos = preds.join(labels_all, on=["pair", "ts"], how="left") if labels_all is not None else preds
        oos = oos.filter((pl.col("ts") >= test_start) & (pl.col("ts") < test_end)).sort(["pair", "ts"])
        if save_to is not None:
            fold_model.save(save_to.format(fold=i))
        folds.append(
            WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                model=fold_model,
                oos=oos,
            )
        )
    return WalkForwardResult(folds=folds)
