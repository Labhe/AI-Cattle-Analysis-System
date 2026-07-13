"""
Regression evaluation for the AI Cattle Analysis System (Phase 14).

Evaluates the weight and BCS regressors with MAE / RMSE / MAPE / R² and
diagnostic plots (actual-vs-predicted, residuals). Reuses the shared
``models.regression.regression_metrics`` so metric definitions stay
consistent with training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from models.regression import regression_metrics


def _new_figure(size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.figure(figsize=size)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path,
                             title: str = "Actual vs Predicted", unit: str = "") -> Path:
    """Scatter of predictions against ground truth with an identity line."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    plt, fig = _new_figure((6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    suffix = f" ({unit})" if unit else ""
    plt.xlabel(f"Actual{suffix}")
    plt.ylabel(f"Predicted{suffix}")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path,
                   title: str = "Residuals") -> Path:
    """Residual (actual − predicted) scatter against predictions."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    plt, fig = _new_figure((7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="none")
    plt.axhline(0, color="r", ls="--", lw=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (actual − predicted)")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    prefix: str = "regressor",
    unit: str = "",
) -> Dict[str, Any]:
    """
    Full regressor evaluation: MAE/RMSE/MAPE/R² + diagnostic plots.

    Returns the metrics dict augmented with a ``plots`` mapping.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = regression_metrics(y_true, y_pred)
    metrics["n_samples"] = int(np.asarray(y_true).ravel().size)

    plots = {
        "actual_vs_predicted": str(plot_actual_vs_predicted(
            y_true, y_pred, output_dir / f"{prefix}_actual_vs_predicted.png",
            title=f"{prefix}: Actual vs Predicted", unit=unit)),
        "residuals": str(plot_residuals(
            y_true, y_pred, output_dir / f"{prefix}_residuals.png",
            title=f"{prefix}: Residuals")),
    }
    metrics["plots"] = plots
    return metrics
