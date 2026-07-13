"""
Weight Regressor Training for the AI Cattle Analysis System (Phase 7).

Estimates live cattle body weight from body measurements extracted from the
segmentation masks (``utils.feature_extraction.extract_body_measurements``)
rather than direct pixel regression. Compares XGBoost, CatBoost, LightGBM,
and Random Forest (whichever are installed), evaluates each with MAE / RMSE /
MAPE / R², selects the best by validation RMSE, and persists the winner with
the metadata needed for confidence scores and prediction intervals.

Training data is built from the unified dataset (``dataset/unify_datasets.py``):
each ``annotations.csv`` row with a valid ``weight_kg`` is paired with its
segmentation mask (``masks/<split>/<stem>.png``); body measurements are
extracted from the mask. A prebuilt measurements CSV can be supplied instead.

Outputs:
  - checkpoints/weight_regressor_best.pkl        selected model + metadata
  - outputs/weight_regressor/model_comparison.csv per-model MAE/RMSE/MAPE/R²
  - outputs/weight_regressor/predictions.csv      test-set actual vs predicted
  - outputs/weight_regressor/training_summary.json
  - logs/train_weight_regressor.log

Usage:
    python -m training.train_weight_regressor
    python -m training.train_weight_regressor --measurements-csv path/to/data.csv
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from models.detection import load_config
from models.regression import WeightRegressor, regression_metrics
from utils.feature_extraction import BODY_MEASUREMENT_NAMES, extract_body_measurements

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"
WEIGHT_COLUMNS = ["weight_kg", "weight", "live_weight", "body_weight"]

logger = logging.getLogger("training.weight_regressor")


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "train_weight_regressor.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def _find_mask(processed_dir: Path, image_rel: str) -> Optional[Path]:
    """Locate the mask PNG for a processed-dataset image path."""
    stem = Path(image_rel).stem
    split = None
    parts = Path(image_rel).parts
    for candidate in ("train", "val", "test"):
        if candidate in parts:
            split = candidate
            break
    search_dirs = [processed_dir / "masks" / split] if split else \
        [processed_dir / "masks" / s for s in ("train", "val", "test")]
    for d in search_dirs:
        mask_path = d / f"{stem}.png"
        if mask_path.exists():
            return mask_path
    return None


def build_measurement_dataset(processed_dir: Path,
                              feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build (X, y, table) from annotations + masks.

    For every annotation row with a valid weight and an existing mask, extract
    body measurements and assemble the feature matrix.
    """
    annotations = processed_dir / "annotations.csv"
    if not annotations.exists():
        raise FileNotFoundError(
            f"Annotations not found: {annotations} (run dataset/unify_datasets.py first)")
    df = pd.read_csv(annotations)
    weight_col = next((c for c in WEIGHT_COLUMNS if c in df.columns), None)
    if weight_col is None:
        raise ValueError(f"No weight column in {annotations} (looked for {WEIGHT_COLUMNS})")

    rows: List[Dict[str, Any]] = []
    df = df[pd.to_numeric(df[weight_col], errors="coerce").notna()]
    for _, row in df.iterrows():
        mask_path = _find_mask(processed_dir, str(row.get("image", "")))
        if mask_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        measurements = extract_body_measurements(mask)
        if measurements["body_area"] <= 0:
            continue
        entry = {name: measurements.get(name, 0.0) for name in feature_names}
        entry["weight_kg"] = float(row[weight_col])
        entry["image"] = row.get("image", "")
        rows.append(entry)

    if not rows:
        raise RuntimeError(
            "No (mask, weight) pairs found. Ensure the dataset has weight labels "
            "and segmentation masks, or pass --measurements-csv.")
    table = pd.DataFrame(rows)
    X = table[feature_names].to_numpy(dtype=float)
    y = table["weight_kg"].to_numpy(dtype=float)
    return X, y, table


def load_measurements_csv(csv_path: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a prebuilt measurements CSV (feature columns + a weight column)."""
    df = pd.read_csv(csv_path)
    weight_col = next((c for c in WEIGHT_COLUMNS if c in df.columns), None)
    if weight_col is None:
        raise ValueError(f"No weight column in {csv_path} (looked for {WEIGHT_COLUMNS})")
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Measurements CSV is missing feature columns: {missing}")
    df = df[pd.to_numeric(df[weight_col], errors="coerce").notna()]
    X = df[feature_names].to_numpy(dtype=float)
    y = pd.to_numeric(df[weight_col]).to_numpy(dtype=float)
    return X, y, df


def train_weight_regressor(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full training + comparison flow; returns the summary dict."""
    config = load_config(args.config)
    paths = config.get("paths", {})
    setup_logging(BASE_DIR / paths.get("logs_dir", "logs"))

    reg_cfg = config.get("weight_regressor", {})
    train_cfg = reg_cfg.get("training", {})
    feature_names = reg_cfg.get("measurement_features", BODY_MEASUREMENT_NAMES + ["heart_girth_sq_length"])
    processed_dir = Path(args.processed_dir) if args.processed_dir \
        else BASE_DIR / paths.get("processed_dir", "data/processed")
    best_path = BASE_DIR / reg_cfg.get("weights", "checkpoints/weight_regressor_best.pkl")
    outputs_dir = BASE_DIR / paths.get("outputs_dir", "outputs") / "weight_regressor"

    logger.info("=" * 60)
    logger.info("  Weight Regressor Training")
    logger.info("=" * 60)
    logger.info(f"Features: {feature_names}")

    if args.measurements_csv:
        X, y, table = load_measurements_csv(Path(args.measurements_csv), feature_names)
        logger.info(f"Loaded {len(X)} samples from {args.measurements_csv}")
    else:
        X, y, table = build_measurement_dataset(processed_dir, feature_names)
        logger.info(f"Built {len(X)} (mask, weight) samples from the processed dataset")

    regressor = WeightRegressor(feature_names=feature_names)
    regressor.fit(X, y, config=config,
                  val_fraction=float(train_cfg.get("val_fraction", 0.2)),
                  seed=int(train_cfg.get("seed", 42)))

    logger.info(f"Model comparison (validation):")
    for name, m in regressor.metrics.items():
        logger.info(f"  {name:<15} MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} "
                    f"MAPE={m['mape']:.2f}% R2={m['r2']:.4f}")
    logger.info(f"Selected best model: {regressor.best_model_name} "
                f"(RMSE={regressor.metrics[regressor.best_model_name]['rmse']:.2f})")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame([
        {"model": name, **m, "selected": name == regressor.best_model_name}
        for name, m in regressor.metrics.items()
    ])
    comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)

    # Full-dataset predictions vs actual for the deployed model.
    preds = regressor.predict(X)
    pd.DataFrame({"actual_kg": y, "predicted_kg": np.round(preds, 1)}).to_csv(
        outputs_dir / "predictions.csv", index=False)
    final_metrics = regression_metrics(y, preds)

    regressor.save(best_path)
    logger.info(f"Saved weight regressor to {best_path}")

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "feature_names": feature_names,
        "best_model": regressor.best_model_name,
        "model_comparison": regressor.metrics,
        "final_metrics": final_metrics,
        "residual_std": regressor.residual_std,
        "checkpoint": str(best_path),
    }
    with open(outputs_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary written to {outputs_dir / 'training_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the cattle weight regressor (config-driven)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override paths.processed_dir (annotations.csv + masks/)")
    parser.add_argument("--measurements-csv", type=str, default=None,
                        help="Prebuilt CSV of body measurements + weight (skips mask extraction)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        train_weight_regressor(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        if logger.handlers:
            logger.error(str(e))
        else:
            print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
