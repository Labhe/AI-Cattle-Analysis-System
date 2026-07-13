"""
Classification evaluation for the AI Cattle Analysis System (Phase 14).

Computes and visualizes confusion matrix, precision / recall / F1 (macro,
weighted, and per-class), per-class accuracy, ROC curves, and
precision-recall curves for the species and breed classifiers.

All plotting uses the matplotlib Agg backend (already a project dependency)
and writes PNGs; metric computation returns plain dicts / arrays so results
can be serialized into the evaluation report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("evaluation.classification")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute the full classification metric set.

    Args:
        y_true: Ground-truth label indices, shape (N,).
        y_pred: Predicted label indices, shape (N,).
        class_names: Ordered class names.
        y_score: Optional predicted probabilities (N, C) for ROC/PR AUC.

    Returns:
        Dict with accuracy, macro/weighted precision-recall-F1, per-class
        metrics, per-class accuracy, the confusion matrix, and (when
        ``y_score`` is given) macro ROC-AUC and average precision.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1)
    per_class_acc = np.divide(cm.diagonal(), row_totals,
                              out=np.zeros(len(class_names)), where=row_totals > 0)

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    metrics: Dict[str, Any] = {
        "n_samples": int(y_true.size),
        "n_classes": len(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "accuracy": float(per_class_acc[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }

    if y_score is not None:
        metrics.update(_auc_metrics(y_true, np.asarray(y_score), labels))
    return metrics


def _auc_metrics(y_true: np.ndarray, y_score: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """Macro one-vs-rest ROC-AUC and average precision (NaN-safe)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=labels)
    if y_bin.shape[1] == 1:  # binary edge case
        y_bin = np.hstack([1 - y_bin, y_bin])
    out: Dict[str, Any] = {}
    try:
        out["roc_auc_macro"] = float(roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr"))
    except ValueError:
        out["roc_auc_macro"] = float("nan")
    try:
        out["average_precision_macro"] = float(average_precision_score(y_bin, y_score, average="macro"))
    except ValueError:
        out["average_precision_macro"] = float("nan")
    return out


# ─────────────────────────────── plots ────────────────────────────────


def _new_figure(size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.figure(figsize=size)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path,
                          normalize: bool = False, title: str = "Confusion Matrix") -> Path:
    """Render a confusion-matrix heatmap to ``out_path``."""
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    plt, fig = _new_figure((max(6, len(class_names) * 0.6),) * 2)
    sns.heatmap(cm, annot=len(class_names) <= 25, fmt=".2f" if normalize else "g",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str],
                    out_path: Path) -> Optional[Path]:
    """One-vs-rest ROC curves (per class + macro) to ``out_path``."""
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    labels = list(range(len(class_names)))
    y_bin = label_binarize(np.asarray(y_true).ravel(), classes=labels)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    y_score = np.asarray(y_score)

    plt, fig = _new_figure((7, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, lw=1.2, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (one-vs-rest)")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_pr_curves(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str],
                   out_path: Path) -> Optional[Path]:
    """One-vs-rest precision-recall curves to ``out_path``."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    labels = list(range(len(class_names)))
    y_bin = label_binarize(np.asarray(y_true).ravel(), classes=labels)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    y_score = np.asarray(y_score)

    plt, fig = _new_figure((7, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        plt.plot(rec, prec, lw=1.2, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (one-vs-rest)")
    plt.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    y_score: Optional[np.ndarray] = None,
    prefix: str = "classifier",
) -> Dict[str, Any]:
    """
    Full classifier evaluation: metrics + all plots written to ``output_dir``.

    Returns the metrics dict augmented with a ``plots`` mapping of artifact
    name -> file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_classification_metrics(y_true, y_pred, class_names, y_score)

    plots: Dict[str, str] = {}
    cm = np.array(metrics["confusion_matrix"])
    plots["confusion_matrix"] = str(plot_confusion_matrix(
        cm, class_names, output_dir / f"{prefix}_confusion_matrix.png"))
    plots["confusion_matrix_normalized"] = str(plot_confusion_matrix(
        cm, class_names, output_dir / f"{prefix}_confusion_matrix_norm.png",
        normalize=True, title="Confusion Matrix (normalized)"))
    if y_score is not None:
        try:
            plots["roc"] = str(plot_roc_curves(y_true, y_score, class_names,
                                               output_dir / f"{prefix}_roc.png"))
            plots["pr"] = str(plot_pr_curves(y_true, y_score, class_names,
                                             output_dir / f"{prefix}_pr.png"))
        except Exception as e:  # noqa: BLE001 — plotting is best-effort
            logger.warning(f"ROC/PR plotting failed: {e}")

    # Per-class accuracy table.
    import pandas as pd
    pd.DataFrame([
        {"class": name, **vals}
        for name, vals in metrics["per_class"].items()
    ]).to_csv(output_dir / f"{prefix}_per_class.csv", index=False)

    metrics["plots"] = plots
    return metrics
