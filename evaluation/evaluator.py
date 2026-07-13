"""
Evaluation orchestrator for the AI Cattle Analysis System (Phase 14).

Ties together classification metrics, regression metrics, benchmarking, and
model comparison into a single evaluation run that writes an
``evaluation_report.json`` plus all supporting artifacts.

Usage (programmatic):
    from evaluation import Evaluator
    ev = Evaluator(output_dir="outputs/evaluation")
    ev.add_classifier("species", y_true, y_pred, class_names, y_score)
    ev.add_regressor("weight", y_true_w, y_pred_w, unit="kg")
    ev.add_benchmark("species_model", model, sample_input, device="cpu")
    report = ev.finalize()

CLI (self-test on synthetic data):
    python -m evaluation.evaluator --demo
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.benchmark import benchmark_model
from evaluation.classification_metrics import evaluate_classifier
from evaluation.model_comparison import compare_models
from evaluation.regression_evaluation import evaluate_regressor

BASE_DIR = Path(__file__).resolve().parents[1]
logger = logging.getLogger("evaluation.evaluator")


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "evaluation.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class Evaluator:
    """
    Accumulates evaluation results across models and writes a unified report.

    Args:
        output_dir: Directory for all evaluation artifacts.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else BASE_DIR / "outputs" / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(BASE_DIR / "logs")
        self.report: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "classifiers": {},
            "regressors": {},
            "benchmarks": {},
            "comparisons": {},
        }

    def add_classifier(self, name: str, y_true: np.ndarray, y_pred: np.ndarray,
                       class_names: List[str], y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate and record a classifier (metrics + confusion/ROC/PR plots)."""
        logger.info(f"Evaluating classifier '{name}' ({len(class_names)} classes)")
        metrics = evaluate_classifier(
            y_true, y_pred, class_names, self.output_dir / name, y_score=y_score, prefix=name)
        self.report["classifiers"][name] = metrics
        logger.info(f"  accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro']['f1']:.4f}")
        return metrics

    def add_regressor(self, name: str, y_true: np.ndarray, y_pred: np.ndarray,
                      unit: str = "") -> Dict[str, Any]:
        """Evaluate and record a regressor (MAE/RMSE/MAPE/R² + plots)."""
        logger.info(f"Evaluating regressor '{name}'")
        metrics = evaluate_regressor(
            y_true, y_pred, self.output_dir / name, prefix=name, unit=unit)
        self.report["regressors"][name] = metrics
        logger.info(f"  MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.4f}")
        return metrics

    def add_benchmark(self, name: str, model: Any, sample_input: Any,
                      device: str = "cpu", n_runs: int = 30) -> Dict[str, Any]:
        """Benchmark a model's inference speed and memory."""
        logger.info(f"Benchmarking '{name}' on {device}")
        result = benchmark_model(model, sample_input, device=device, n_runs=n_runs)
        self.report["benchmarks"][name] = result
        logger.info(f"  latency={result['latency_ms']['mean']:.2f}ms "
                    f"throughput={result['throughput_per_sec']:.1f}/s")
        return result

    def add_comparison(self, name: str, results: Dict[str, Dict[str, Any]],
                       metric_keys: List[str], primary_metric: str,
                       higher_is_better: bool = True) -> Dict[str, Any]:
        """Build a model-comparison report from collected metrics."""
        comparison = compare_models(
            results, metric_keys, primary_metric, self.output_dir / "comparisons",
            higher_is_better=higher_is_better, prefix=name)
        self.report["comparisons"][name] = comparison
        logger.info(f"Comparison '{name}': best = {comparison['best_model']}")
        return comparison

    def finalize(self) -> Dict[str, Any]:
        """Write the unified evaluation report JSON and return it."""
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Evaluation report written to {report_path}")
        self.report["report_path"] = str(report_path)
        return self.report


def _run_demo(output_dir: Path) -> Dict[str, Any]:
    """Self-test the framework on synthetic data (no trained models needed)."""
    import torch
    from models.species_classifier import build_species_classifier

    rng = np.random.default_rng(0)
    ev = Evaluator(output_dir=output_dir)

    # Classification demo (5 classes, correlated predictions + scores).
    classes = ["Cow", "Buffalo", "Goat", "Sheep", "Horse"]
    n = 200
    y_true = rng.integers(0, 5, n)
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.2
    y_pred[flip] = rng.integers(0, 5, flip.sum())
    y_score = rng.random((n, 5))
    y_score[np.arange(n), y_true] += 1.5  # make the true class likely
    y_score = y_score / y_score.sum(axis=1, keepdims=True)
    ev.add_classifier("species", y_true, y_pred, classes, y_score)

    # Regression demo.
    yt = rng.uniform(200, 800, 150)
    yp = yt + rng.normal(0, 25, 150)
    ev.add_regressor("weight", yt, yp, unit="kg")
    bt = np.clip(np.round(rng.uniform(1, 5, 150) * 2) / 2, 1, 5)
    bp = np.clip(bt + rng.normal(0, 0.3, 150), 1, 5)
    ev.add_regressor("bcs", bt, bp)

    # Benchmark demo (tiny model, CPU).
    model = build_species_classifier("efficientnet_v2_s", num_classes=5, pretrained=False)
    ev.add_benchmark("species_model", model, torch.randn(1, 3, 224, 224),
                     device="cpu", n_runs=5)

    # Comparison demo across the regressor metrics.
    ev.add_comparison(
        "regressors",
        {"weight": ev.report["regressors"]["weight"],
         "bcs": ev.report["regressors"]["bcs"]},
        metric_keys=["mae", "rmse", "mape", "r2"],
        primary_metric="r2", higher_is_better=True)

    return ev.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Cattle Analysis evaluation framework")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic self-test of the framework")
    parser.add_argument("--output-dir", type=Path,
                        default=BASE_DIR / "outputs" / "evaluation")
    args = parser.parse_args()

    if args.demo:
        report = _run_demo(args.output_dir)
        print(json.dumps({
            "classifiers": {k: v["accuracy"] for k, v in report["classifiers"].items()},
            "regressors": {k: v["r2"] for k, v in report["regressors"].items()},
            "benchmarks": {k: v["latency_ms"]["mean"] for k, v in report["benchmarks"].items()},
            "report_path": report["report_path"],
        }, indent=2))
        return 0

    print("Nothing to do. Use --demo to self-test, or import Evaluator programmatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
