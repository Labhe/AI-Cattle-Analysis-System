"""
BCS Regressor Training for the AI Cattle Analysis System (Phase 8).

Trains a dedicated Body Condition Score regressor that predicts BCS on the
standard 1–5 scale in 0.5 steps (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5) from
morphometric + body-measurement features derived from the segmentation mask.
Compares XGBoost / CatBoost / LightGBM / Random Forest, selects the best by
validation RMSE, and reports MAE / RMSE / MAPE / R² plus discrete BCS
accuracy (exact grid match) and within-0.5 accuracy.

Training data is built from the unified dataset: each ``annotations.csv`` row
with a valid ``body_condition_score`` is paired with its segmentation mask;
features are extracted from the mask. A prebuilt features CSV can be supplied.

Outputs:
  - checkpoints/bcs_tree_best.pkl              selected model + metadata
  - outputs/bcs_regressor/model_comparison.csv per-model MAE/RMSE/MAPE/R²
  - outputs/bcs_regressor/predictions.csv       actual vs predicted BCS
  - outputs/bcs_regressor/training_summary.json
  - logs/train_bcs_regressor.log

Usage:
    python -m training.train_bcs_regressor
    python -m training.train_bcs_regressor --features-csv path/to/data.csv
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
from models.regression import BCSRegressor, regression_metrics, snap_to_bcs
from utils.feature_extraction import extract_body_measurements, extract_features_from_mask

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"
BCS_COLUMNS = ["body_condition_score", "bcs", "bcs_score", "condition_score"]

logger = logging.getLogger("training.bcs_regressor")


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "train_bcs_regressor.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def _extract_all_features(mask: np.ndarray) -> Dict[str, float]:
    """Union of morphometric + body-measurement features for a mask."""
    feats = extract_features_from_mask(mask)
    feats.update(extract_body_measurements(mask))
    return feats


def _find_mask(processed_dir: Path, image_rel: str) -> Optional[Path]:
    """Locate the mask PNG for a processed-dataset image path."""
    stem = Path(image_rel).stem
    parts = Path(image_rel).parts
    split = next((c for c in ("train", "val", "test") if c in parts), None)
    search_dirs = [processed_dir / "masks" / split] if split else \
        [processed_dir / "masks" / s for s in ("train", "val", "test")]
    for d in search_dirs:
        mask_path = d / f"{stem}.png"
        if mask_path.exists():
            return mask_path
    return None


def build_bcs_dataset(processed_dir: Path,
                      feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build (X, y, table) from annotations with BCS labels + masks."""
    annotations = processed_dir / "annotations.csv"
    if not annotations.exists():
        raise FileNotFoundError(
            f"Annotations not found: {annotations} (run dataset/unify_datasets.py first)")
    df = pd.read_csv(annotations)
    bcs_col = next((c for c in BCS_COLUMNS if c in df.columns), None)
    if bcs_col is None:
        raise ValueError(f"No BCS column in {annotations} (looked for {BCS_COLUMNS})")

    df = df[pd.to_numeric(df[bcs_col], errors="coerce").notna()]
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        mask_path = _find_mask(processed_dir, str(row.get("image", "")))
        if mask_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        feats = _extract_all_features(mask)
        if feats.get("body_area", feats.get("area", 0)) <= 0:
            continue
        entry = {name: feats.get(name, 0.0) for name in feature_names}
        entry["bcs"] = float(row[bcs_col])
        rows.append(entry)

    if not rows:
        raise RuntimeError(
            "No (mask, BCS) pairs found. Ensure the dataset has BCS labels and "
            "segmentation masks, or pass --features-csv.")
    table = pd.DataFrame(rows)
    X = table[feature_names].to_numpy(dtype=float)
    y = table["bcs"].to_numpy(dtype=float)
    return X, y, table


def load_features_csv(csv_path: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a prebuilt features CSV (feature columns + a BCS column)."""
    df = pd.read_csv(csv_path)
    bcs_col = next((c for c in BCS_COLUMNS if c in df.columns), None)
    if bcs_col is None:
        raise ValueError(f"No BCS column in {csv_path} (looked for {BCS_COLUMNS})")
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Features CSV is missing columns: {missing}")
    df = df[pd.to_numeric(df[bcs_col], errors="coerce").notna()]
    X = df[feature_names].to_numpy(dtype=float)
    y = pd.to_numeric(df[bcs_col]).to_numpy(dtype=float)
    return X, y, df


def train_bcs_regressor(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full training + comparison flow; returns the summary dict."""
    config = load_config(args.config)
    paths = config.get("paths", {})
    setup_logging(BASE_DIR / paths.get("logs_dir", "logs"))

    bcs_cfg = config.get("bcs_regressor", {})
    train_cfg = bcs_cfg.get("training", {})
    feature_names = bcs_cfg.get("features",
                                ["solidity", "extent", "compactness", "aspect_ratio",
                                 "chest_width", "heart_girth", "body_area"])
    processed_dir = Path(args.processed_dir) if args.processed_dir \
        else BASE_DIR / paths.get("processed_dir", "data/processed")
    best_path = BASE_DIR / bcs_cfg.get("tree_weights", "checkpoints/bcs_tree_best.pkl")
    outputs_dir = BASE_DIR / paths.get("outputs_dir", "outputs") / "bcs_regressor"

    logger.info("=" * 60)
    logger.info("  BCS Regressor Training")
    logger.info("=" * 60)
    logger.info(f"Features: {feature_names}")

    if args.features_csv:
        X, y, table = load_features_csv(Path(args.features_csv), feature_names)
        logger.info(f"Loaded {len(X)} samples from {args.features_csv}")
    else:
        X, y, table = build_bcs_dataset(processed_dir, feature_names)
        logger.info(f"Built {len(X)} (mask, BCS) samples from the processed dataset")

    scale_cfg = bcs_cfg.get("scale", [1.0, 5.0])
    step = float(bcs_cfg.get("step", 0.5))
    scale = list(np.arange(scale_cfg[0], scale_cfg[-1] + step / 2, step))

    regressor = BCSRegressor(feature_names=feature_names, scale=scale)
    regressor.fit(X, y, config=config,
                  val_fraction=float(train_cfg.get("val_fraction", 0.2)),
                  seed=int(train_cfg.get("seed", 42)))

    logger.info("Model comparison (validation):")
    for name, m in regressor.metrics.items():
        logger.info(f"  {name:<15} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
                    f"MAPE={m['mape']:.2f}% R2={m['r2']:.4f}")
    logger.info(f"Selected best model: {regressor.best_model_name}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": name, **m, "selected": name == regressor.best_model_name}
        for name, m in regressor.metrics.items()
    ]).to_csv(outputs_dir / "model_comparison.csv", index=False)

    preds_snapped = regressor.predict(X, snap=True)
    exact = float(np.mean(preds_snapped == np.array([snap_to_bcs(v, scale) for v in y])))
    within_half = float(np.mean(np.abs(preds_snapped - y) <= step))
    pd.DataFrame({"actual_bcs": y, "predicted_bcs": preds_snapped}).to_csv(
        outputs_dir / "predictions.csv", index=False)
    final_metrics = regression_metrics(y, regressor.predict(X, snap=False))
    logger.info(f"Exact BCS accuracy={exact:.3f} | within-{step} accuracy={within_half:.3f}")

    regressor.save(best_path)
    logger.info(f"Saved BCS regressor to {best_path}")

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "feature_names": feature_names,
        "scale": scale,
        "best_model": regressor.best_model_name,
        "model_comparison": regressor.metrics,
        "final_metrics": final_metrics,
        "exact_accuracy": exact,
        "within_half_accuracy": within_half,
        "uncertainty": regressor.residual_std,
        "checkpoint": str(best_path),
    }
    with open(outputs_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary written to {outputs_dir / 'training_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BCS regressor (config-driven)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override paths.processed_dir (annotations.csv + masks/)")
    parser.add_argument("--features-csv", type=str, default=None,
                        help="Prebuilt CSV of features + BCS (skips mask extraction)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        train_bcs_regressor(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        if logger.handlers:
            logger.error(str(e))
        else:
            print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
