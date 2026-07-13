"""
Evaluation module for the AI Cattle Analysis System (Phase 14).

Public API for classification / regression evaluation, benchmarking, model
comparison, and the unified :class:`Evaluator` orchestrator.
"""

from evaluation.benchmark import benchmark_inference, benchmark_model
from evaluation.classification_metrics import (
    compute_classification_metrics,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
)
from evaluation.evaluator import Evaluator
from evaluation.model_comparison import compare_models
from evaluation.regression_evaluation import evaluate_regressor

__all__ = [
    "Evaluator",
    "compute_classification_metrics",
    "evaluate_classifier",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    "evaluate_regressor",
    "benchmark_inference",
    "benchmark_model",
    "compare_models",
]
