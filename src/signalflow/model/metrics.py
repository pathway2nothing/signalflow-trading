"""Classification-quality metrics computed on a model's leak-free OOS predictions."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target import LABEL_COL

METRICS = ("f1", "precision", "recall", "pr_auc", "roc_auc", "brier")
"""Metric columns of :func:`scorecard_table`; ``auc`` is accepted as an alias of ``roc_auc``."""


def _joined(model, data: Dataset) -> pl.DataFrame:
    preds = model.predict_oos(data).drop_nulls(subset=[model.output])
    labels = model.target.labels(data)
    return preds.join(labels, on=["pair", "ts"], how="inner").drop_nulls(subset=[LABEL_COL])


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """The six metrics on binary labels ``y`` and scores ``p`` at a firing ``threshold``."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    single_class = len(np.unique(y)) < 2
    binary = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, binary, average="binary", zero_division=0.0)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pr_auc": float(average_precision_score(y, p)) if not single_class else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if not single_class else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
    }


def resolve_operating(operating: "float | str", train_scores: "pl.Series | None", scores: pl.Series) -> float:
    """A firing threshold from an operating-point spec.

    ``0.6`` - a fixed threshold; ``"q0.9"`` - that quantile of the evaluated (OOS)
    scores; ``"train_q0.9"`` - that quantile of the *training-window* scores, the
    leak-safe choice (falls back to the OOS quantile when no train scores exist).
    """
    if not isinstance(operating, str):
        return float(operating)
    spec = operating.strip().lower()
    if spec.startswith("train_q"):
        q = float(spec[len("train_q") :])
        pool = train_scores if train_scores is not None and train_scores.len() else scores
    elif spec.startswith("q"):
        q = float(spec[1:])
        pool = scores
    else:
        raise ValueError(f"operating must be a threshold, 'q<quantile>' or 'train_q<quantile>', got {operating!r}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"operating quantile must lie in [0, 1], got {q}")
    return float(pool.drop_nulls().quantile(q))


def classification_scorecard(model, data: Dataset, threshold: "float | str" = 0.5) -> dict:
    """AUC / PR-AUC / Brier / precision / recall / F1 of ``predict_oos`` vs the target labels.

    ``threshold`` is an operating point as in :func:`resolve_operating` (a number,
    ``"q0.9"`` over the OOS scores, or ``"train_q0.9"`` over in-sample scores).
    Labels are treated as binary (any positive class vs the rest); multi-class
    targets should be binarized before use.
    """
    df = _joined(model, data)
    y = df.get_column(LABEL_COL).cast(pl.Int8).to_numpy()
    scores = df.get_column(model.output)
    train_scores = None
    if isinstance(threshold, str) and threshold.lower().startswith("train_q"):
        train_scores = model.predict(data).get_column(model.output)
    thr = resolve_operating(threshold, train_scores, scores)
    p = scores.to_numpy()
    m = _binary_metrics(y, p, thr)
    return {
        "n": int(df.height),
        "threshold": thr,
        "base_rate": float(y.mean()) if y.size else 0.0,
        "auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "brier": m["brier"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
    }


def _target_params(model) -> str:
    try:
        params = model.target.to_config().get("params", {})
    except Exception:
        return ""
    return ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in params.items())


def _row(name: str, model, y: np.ndarray, p: np.ndarray, thr: float, metrics: Sequence[str], extra: dict) -> dict:
    row = {
        "model": name,
        "target": getattr(model.target, "name", type(model.target).__name__),
        "target_params": _target_params(model),
        **extra,
        "n_test": int(y.size),
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "threshold": thr,
    }
    values = _binary_metrics(y, p, thr) if y.size else dict.fromkeys(METRICS, float("nan"))
    for m in metrics:
        row[m] = values["roc_auc" if m == "auc" else m]
    return row


def scorecard_table(
    models: Any,
    data: Dataset,
    operating: "float | str" = 0.5,
    metrics: Sequence[str] = METRICS,
    round_to: "int | None" = 4,
) -> pl.DataFrame:
    """One scorecard row per model (or per walk-forward fold), ready to compare and average.

    ``models`` is a ``ForecastModel``, a list of them, a ``{name: model}`` mapping,
    or a :class:`~signalflow.model.walkforward.WalkForwardResult`. Models are
    scored on ``predict_oos(data)`` against their target's labels; walk-forward
    folds on their own OOS window, with ``"train_q<q>"`` operating points taken
    from the fold model's scores over its training window (what a live threshold
    calibrated on the past would have been). Columns: ``model``, ``target``,
    ``target_params``, ``n_test``, ``prevalence``, ``threshold`` and ``metrics``
    (any of :data:`METRICS`, ``auc`` = ``roc_auc``); fold rows add ``fold``,
    ``tag``, ``test_start``, ``test_end``. Use :func:`scorecard_means` to average
    by target.
    """
    unknown = [m for m in metrics if m not in (*METRICS, "auc")]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; choose from {list(METRICS)}")
    rows: list[dict] = []
    folds = getattr(models, "folds", None)
    if folds is not None:
        for i, fold in enumerate(folds):
            model = fold.model
            frame = fold.oos.drop_nulls(subset=[model.output, LABEL_COL]) if fold.oos is not None else None
            if frame is None or frame.height == 0:
                continue
            train_scores = None
            if isinstance(operating, str) and operating.lower().startswith("train_q"):
                # Score only the fold's own training window: predicting the whole dataset
                # per fold would cost as much again as the walk-forward that produced it.
                train_ds = data.slice_time(fold.train_start, fold.test_start)
                train_scores = model.predict(train_ds).get_column(model.output)
            thr = resolve_operating(operating, train_scores, frame.get_column(model.output))
            y = frame.get_column(LABEL_COL).cast(pl.Int8).to_numpy()
            p = frame.get_column(model.output).to_numpy()
            extra = {"fold": i, "tag": fold.tag, "test_start": fold.test_start, "test_end": fold.test_end}
            rows.append(_row(f"fold{i}", model, y, p, thr, metrics, extra))
    else:
        named: list[tuple[str, Any]]
        if isinstance(models, Mapping):
            named = list(models.items())
        elif isinstance(models, (list, tuple)):
            named = [(f"{getattr(m.target, 'name', 'model')}#{i}", m) for i, m in enumerate(models)]
        else:
            named = [(getattr(models.target, "name", "model"), models)]
        for name, model in named:
            df = _joined(model, data)
            train_scores = None
            if isinstance(operating, str) and operating.lower().startswith("train_q"):
                train_scores = model.predict(data).get_column(model.output)
            thr = resolve_operating(operating, train_scores, df.get_column(model.output))
            y = df.get_column(LABEL_COL).cast(pl.Int8).to_numpy()
            p = df.get_column(model.output).to_numpy()
            rows.append(_row(name, model, y, p, thr, metrics, {}))
    if not rows:
        return pl.DataFrame(schema={"model": pl.Utf8, "target": pl.Utf8, "n_test": pl.Int64})
    table = pl.DataFrame(rows)
    if round_to is not None:
        float_cols = [c for c in table.columns if table.get_column(c).dtype in (pl.Float32, pl.Float64)]
        table = table.with_columns([pl.col(c).round(round_to) for c in float_cols])
    return table


def scorecard_means(
    table: pl.DataFrame, by: "str | Sequence[str]" = "target", round_to: "int | None" = 4
) -> pl.DataFrame:
    """Average the metric columns of a :func:`scorecard_table` per ``by`` (default: per target), with row counts."""
    keys = [by] if isinstance(by, str) else list(by)
    metric_cols = [c for c in (*METRICS, "auc") if c in table.columns]
    aggs = [pl.len().alias("rows"), pl.col("n_test").sum().alias("n_test")] + [
        pl.col(c).mean().alias(c) for c in metric_cols
    ]
    out = table.group_by(keys, maintain_order=True).agg(aggs).sort(keys)
    if round_to is not None:
        out = out.with_columns([pl.col(c).round(round_to) for c in metric_cols])
    return out


__all__ = ["METRICS", "classification_scorecard", "resolve_operating", "scorecard_means", "scorecard_table"]
