from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from signalflow.analytic.base import SignalMetric
from signalflow.core import RawData, Signals, sf_component


@dataclass
@sf_component(name="classification")
class SignalClassificationMetric(SignalMetric):
    """Analyze signal classification performance against labels.

    Computes standard classification metrics including:
    - Precision, Recall, F1 Score
    - Confusion Matrix
    - ROC Curve and AUC
    - Signal strength distribution

    Requires labels to be provided.
    """

    positive_labels: list = field(default_factory=list)
    negative_labels: list = field(default_factory=list)

    chart_height: int = 900
    chart_width: int = 1400
    roc_n_thresholds: int = 100

    def __post_init__(self):
        """Set default label mappings if not provided."""
        if not self.positive_labels:
            self.positive_labels = ["rise", "up", 1, "positive", "buy"]
        if not self.negative_labels:
            self.negative_labels = ["fall", "down", 0, "negative", "sell"]

    def _map_labels_to_binary(self, labels: np.ndarray) -> np.ndarray:
        """Convert string/mixed labels to binary (0/1).

        Args:
            labels: Array of labels (can be strings, ints, etc.)

        Returns:
            Binary numpy array (0 for negative, 1 for positive)
        """
        binary_labels = np.zeros(len(labels), dtype=int)

        for i, label in enumerate(labels):
            if label in self.positive_labels:
                binary_labels[i] = 1
            elif label in self.negative_labels:
                binary_labels[i] = 0
            else:
                logger.warning(f"Unknown label value: {label}, treating as negative")
                binary_labels[i] = 0

        return binary_labels

    def compute(
        self,
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute classification metrics."""

        if labels is None:
            logger.error("Labels are required for classification metrics")
            return None, {}

        signals_df = signals.value

        signals_with_labels = signals_df.join(labels, on=["timestamp", "pair"], how="inner")

        predictions = signals_with_labels.filter(pl.col("signal") != 0)

        if predictions.height == 0:
            logger.warning("No non-zero signals found for classification")
            return None, {}

        logger.info(f"Found {predictions.height} signal-label pairs for classification")

        y_pred = predictions["signal"].to_numpy()
        y_true_raw = predictions["label"].to_numpy()
        unique_labels = np.unique(y_true_raw)
        logger.info(f"Unique label values: {unique_labels}")
        logger.info(f"Unique prediction values: {np.unique(y_pred)}")

        y_true = self._map_labels_to_binary(y_true_raw)

        y_pred_binary = (y_pred > 0).astype(int)

        logger.info(f"After conversion - Unique y_true: {np.unique(y_true)}, y_pred: {np.unique(y_pred_binary)}")

        if "strength" in predictions.columns:
            strengths = predictions["strength"].to_numpy()
        else:
            strengths = np.abs(y_pred).astype(float)

        if np.std(strengths) < 1e-10:
            logger.warning("All strengths are identical, ROC curve will be degenerate")
            roc_scores = y_pred_binary.astype(float)
        else:
            roc_scores = (strengths - strengths.min()) / (strengths.max() - strengths.min())

        try:
            cm = confusion_matrix(y_true, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")
                tn, fp, fn, tp = 1, 1, 1, 1
        except Exception as e:
            logger.warning(f"Could not compute confusion matrix: {e}, using defaults")
            tn, fp, fn, tp = 1, 1, 1, 1

        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2

        positive_rate = np.mean(y_true)

        if len(np.unique(y_true)) > 1:
            try:
                fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_true, roc_scores)
                auc = roc_auc_score(y_true, roc_scores)
            except Exception as e:
                logger.warning(f"Could not compute ROC: {e}")
                fpr_arr, tpr_arr, thresholds_arr = np.array([0, 1]), np.array([0, 1]), np.array([1, 0])
                auc = 0.5
        else:
            fpr_arr, tpr_arr, thresholds_arr = np.array([0, 1]), np.array([0, 1]), np.array([1, 0])
            auc = 0.5
            logger.warning("Only one class present, AUC undefined")

        logloss = np.nan
        try:
            if len(np.unique(y_true)) > 1 and np.std(roc_scores) > 1e-10:
                probs = np.clip(roc_scores, 1e-10, 1 - 1e-10)
                logloss = log_loss(y_true=y_true, y_pred=probs, labels=[0, 1])
        except Exception as e:
            logger.warning(f"Could not compute log loss: {e}")

        strength_mean = float(np.mean(strengths))
        strength_std = float(np.std(strengths)) if len(strengths) > 1 else 0.0
        strength_quartiles = np.percentile(strengths, [25, 50, 75]).tolist() if len(strengths) > 0 else [0, 0, 0]

        computed_metrics = {
            "quant": {
                "total_signals": int(predictions.height),
                "total_positive_signals": int(tp + fp),
                "total_negative_signals": int(tn + fn),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "sensitivity": float(sensitivity),
                "balanced_accuracy": float(balanced_acc),
                "f1": float(f1),
                "positive_rate": float(positive_rate),
                "auc": float(auc),
                "log_loss": float(logloss) if not np.isnan(logloss) else None,
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "strength_mean": strength_mean,
                "strength_std": strength_std,
            },
            "series": {
                "roc_curve": {
                    "tpr": tpr_arr.tolist(),
                    "fpr": fpr_arr.tolist(),
                    "thresholds": thresholds_arr.tolist(),
                },
                "strength_quartiles": strength_quartiles,
                "strengths_raw": strengths.tolist(),
            },
        }

        plots_context = {
            "total_samples": predictions.height,
            "label_mapping": {
                "positive": self.positive_labels,
                "negative": self.negative_labels,
            },
        }

        logger.info(
            f"Classification metrics computed: "
            f"Precision={precision:.3f}, Recall={recall:.3f}, "
            f"F1={f1:.3f}, AUC={auc:.3f}"
        )

        return computed_metrics, plots_context

    def plot(
        self,
        computed_metrics: dict[str, Any],
        plots_context: dict[str, Any],
        raw_data: RawData,
        signals: Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate classification metrics visualization."""

        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None

        fig = self._create_figure()

        self._add_roc_curve(fig, computed_metrics)
        self._add_confusion_matrix(fig, computed_metrics)
        self._add_strength_distribution(fig, computed_metrics)
        self._add_metrics_table(fig, computed_metrics)
        self._update_layout(fig)

        return fig

    @staticmethod
    def _create_figure():
        """Create subplot structure."""
        return make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "ROC Curve",
                "Confusion Matrix",
                "Signal Strength Distribution",
                "Key Metrics",
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

    @staticmethod
    def _add_confusion_matrix(fig, metrics):
        """Add confusion matrix heatmap."""
        cm = metrics["quant"]["confusion_matrix"]
        cm_values = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        total = sum(cm.values())
        cm_pcts = [[val / total * 100 if total > 0 else 0 for val in row] for row in cm_values]

        fig.add_trace(
            go.Heatmap(
                z=cm_values,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale=[
                    [0, "#f0f9e8"],
                    [0.5, "#bae4bc"],
                    [1, "#7bccc4"],
                ],
                showscale=False,
                hovertemplate=("<b>%{y} / %{x}</b><br>Count: %{z}<br><extra></extra>"),
            ),
            row=1,
            col=2,
        )

        for i, actual in enumerate(["Actual Negative", "Actual Positive"]):
            for j, predicted in enumerate(["Predicted Negative", "Predicted Positive"]):
                fig.add_annotation(
                    text=f"<b>{cm_values[i][j]}</b><br>({cm_pcts[i][j]:.1f}%)",
                    x=predicted,
                    y=actual,
                    showarrow=False,
                    font=dict(
                        color="#1a1a1a",
                        size=14,
                    ),
                    row=1,
                    col=2,
                )

    @staticmethod
    def _add_roc_curve(fig, metrics):
        """Add ROC curve plot."""
        roc = metrics["series"]["roc_curve"]
        auc = metrics["quant"]["auc"]

        fig.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name="ROC Curve",
                line=dict(color="#2171b5", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(33, 113, 181, 0.1)",
                hovertemplate=("<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="#969696", dash="dash", width=1.5),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            x=0.6,
            y=0.15,
            text=f"<b>AUC = {auc:.3f}</b>",
            showarrow=False,
            font=dict(size=14, color="#2171b5"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#2171b5",
            borderwidth=1,
            borderpad=6,
            row=1,
            col=1,
        )

    @staticmethod
    def _add_strength_distribution(fig, metrics):
        """Add signal strength distribution plot."""
        if "strengths_raw" in metrics["series"]:
            strengths = np.array(metrics["series"]["strengths_raw"])
        else:
            mean = metrics["quant"]["strength_mean"]
            std = metrics["quant"]["strength_std"]
            if std < 1e-10:
                std = 0.01
            strengths = np.random.normal(mean, std, 1000)

        quartiles = metrics["series"]["strength_quartiles"]

        fig.add_trace(
            go.Histogram(
                x=strengths,
                name="Strength Distribution",
                nbinsx=30,
                marker_color="#74c476",
                marker_line_color="#238b45",
                marker_line_width=1,
                opacity=0.8,
                histnorm="probability density",
                hovertemplate=("<b>Strength:</b> %{x:.3f}<br><b>Density:</b> %{y:.4f}<extra></extra>"),
            ),
            row=2,
            col=1,
        )

        if len(set(quartiles)) > 1:
            quartile_colors = ["#d94801", "#8856a7", "#d94801"]
            quartile_names = ["Q1", "Median", "Q3"]

            for q_val, color, name in zip(quartiles, quartile_colors, quartile_names, strict=False):
                fig.add_vline(
                    x=q_val,
                    line_color=color,
                    line_dash="dash",
                    line_width=1.5,
                    annotation_text=name,
                    annotation_position="top",
                    annotation_font_size=10,
                    annotation_font_color=color,
                    row=2,
                    col=1,
                )

    @staticmethod
    def _add_metrics_table(fig, metrics):
        """Add metrics summary table."""
        quant = metrics["quant"]

        table_data = [
            ["Total Signals", f"{quant['total_signals']}"],
            ["Positive Signals", f"{quant['total_positive_signals']}"],
            ["Precision", f"{quant['precision']:.3f}"],
            ["Recall", f"{quant['recall']:.3f}"],
            ["Specificity", f"{quant['specificity']:.3f}"],
            ["Sensitivity", f"{quant['sensitivity']:.3f}"],
            ["F1 Score", f"{quant['f1']:.3f}"],
            ["Balanced Accuracy", f"{quant['balanced_accuracy']:.3f}"],
            ["Positive Rate", f"{quant['positive_rate']:.3f}"],
            ["Mean Strength", f"{quant['strength_mean']:.3f}"],
            ["Strength Std", f"{quant['strength_std']:.3f}"],
        ]

        if quant["log_loss"] is not None:
            table_data.append(["Log Loss", f"{quant['log_loss']:.3f}"])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="#e6f2ff",
                    align="left",
                    font=dict(size=12, color="#084594"),
                    line_color="#b3d4ff",
                    height=32,
                ),
                cells=dict(
                    values=list(zip(*table_data, strict=False)),
                    fill_color="#f7fbff",
                    align="left",
                    font=dict(size=11, color="#333333"),
                    line_color="#e6f2ff",
                    height=26,
                ),
            ),
            row=2,
            col=2,
        )

    def _update_layout(self, fig):
        """Update figure layout and axes."""
        fig.update_xaxes(
            title_text="False Positive Rate",
            range=[0, 1],
            gridcolor="#e0e0e0",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="True Positive Rate",
            range=[0, 1],
            gridcolor="#e0e0e0",
            row=1,
            col=1,
        )

        fig.update_xaxes(
            title_text="Predicted Class",
            row=1,
            col=2,
        )
        fig.update_yaxes(
            title_text="Actual Class",
            row=1,
            col=2,
        )

        fig.update_xaxes(
            title_text="Signal Strength",
            gridcolor="#e0e0e0",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Probability Density",
            gridcolor="#e0e0e0",
            row=2,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text="<b>Classification Performance Analysis</b><br>"
                "<sup>ROC Curve · Confusion Matrix · Strength Distribution</sup>",
                font=dict(color="#333333", size=18),
                x=0.5,
                xanchor="center",
            ),
            height=self.chart_height,
            width=self.chart_width,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e0e0e0",
                borderwidth=1,
            ),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#ffffff",
        )
