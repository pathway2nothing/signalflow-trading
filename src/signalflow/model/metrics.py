"""Classification-quality metrics computed on a model's leak-free OOS predictions."""

import numpy as np
import polars as pl

from signalflow.data.dataset import Dataset
from signalflow.target import LABEL_COL


def _joined(model, data: Dataset) -> pl.DataFrame:
    preds = model.predict_oos(data).drop_nulls(subset=[model.output])
    labels = model.target.labels(data)
    return preds.join(labels, on=["pair", "ts"], how="inner").drop_nulls(subset=[LABEL_COL])


def classification_scorecard(model, data: Dataset, threshold: float = 0.5) -> dict:
    """AUC / PR-AUC / Brier / precision / recall / F1 of ``predict_oos`` vs the target labels.

    Labels are treated as binary (any positive class vs the rest); multi-class
    targets should be binarized before use.
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    df = _joined(model, data)
    y = df.get_column(LABEL_COL).cast(pl.Int8).to_numpy()
    p = df.get_column(model.output).to_numpy()
    binary = (p >= threshold).astype(int)
    single_class = len(np.unique(y)) < 2
    precision, recall, f1, _ = precision_recall_fscore_support(y, binary, average="binary", zero_division=0.0)
    return {
        "n": int(df.height),
        "threshold": threshold,
        "base_rate": float(y.mean()) if y.size else 0.0,
        "auc": float(roc_auc_score(y, p)) if not single_class else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if not single_class else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
