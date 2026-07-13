"""
Segmentor Training for the AI Cattle Analysis System (Phase 4).

Fine-tunes an Ultralytics YOLO segmentation model (YOLO11-seg when supported,
YOLOv8-seg otherwise) on a YOLO segmentation dataset (polygon labels). All
hyperparameters come from ``configs/model_config.yaml`` (``segmentation``
section); CLI flags override individual values for experiments.

To train body-part segmentation (animal / head / torso / legs / tail), point
``--data`` at a dataset whose ``data.yaml`` declares those classes — see
``segmentation.classes`` in the config.

Outputs:
  - checkpoints/segmentor_best.pt      best checkpoint (used by inference.py)
  - checkpoints/segmentor_last.pt      last checkpoint (resume source)
  - runs/segment/<run-name>/           Ultralytics run directory (plots, curves)
  - outputs/segmentation/metrics_<split>.csv    validation metrics per split
  - outputs/segmentation/training_summary.json  run summary + final metrics
  - outputs/segmentation/visualizations/        sample mask overlays + polygons
  - logs/train_segmentor.log           training log

Usage:
    python -m training.train_segmentor
    python -m training.train_segmentor --model-size s --epochs 50 --batch 8
    python -m training.train_segmentor --resume
    python -m training.train_segmentor --validate-only --weights checkpoints/segmentor_best.pt
"""

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import yaml

from models.detection import MODEL_SIZES, load_config, resolve_model_name, select_device
from models.segmentation import LivestockSegmentor, export_segmentation

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"
DEFAULT_RUN_NAME = "livestock_segmentor"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger("training.segmentor")


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "train_segmentor.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def resolve_data_yaml(config: Dict[str, Any], override: Optional[Path]) -> Path:
    """
    Locate the segmentation dataset descriptor: the CLI override, else a
    dedicated ``data/segmentation/data.yaml``, else the unified dataset.
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--data file does not exist: {override}")
        return override

    processed_dir = BASE_DIR / config.get("paths", {}).get("processed_dir", "data/processed")
    candidates = [BASE_DIR / "data" / "segmentation" / "data.yaml",
                  processed_dir / "data.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No dataset descriptor found. Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + ". Provide a YOLO segmentation dataset (polygon labels) via --data."
    )


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map non-None CLI flags onto Ultralytics train-argument names."""
    mapping = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
               "patience": args.patience, "lr0": args.lr, "workers": args.workers}
    return {k: v for k, v in mapping.items() if v is not None}


def find_resume_checkpoint(runs_dir: Path, run_name: str, config: Dict[str, Any]) -> Path:
    """Locate the last.pt to resume from (run directory first, then checkpoints)."""
    candidates = [runs_dir / run_name / "weights" / "last.pt"]
    last_cfg = config.get("segmentation", {}).get("last_checkpoint")
    if last_cfg:
        candidates.append(BASE_DIR / last_cfg)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No checkpoint to resume from. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def save_checkpoints(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Copy best/last run weights to the stable checkpoint paths from the config."""
    seg_cfg = config.get("segmentation", {})
    saved: Dict[str, Optional[str]] = {"best": None, "last": None}
    targets = {
        "best": BASE_DIR / seg_cfg.get("fine_tuned_weights", "checkpoints/segmentor_best.pt"),
        "last": BASE_DIR / seg_cfg.get("last_checkpoint", "checkpoints/segmentor_last.pt"),
    }
    for kind, target in targets.items():
        source = run_dir / "weights" / f"{kind}.pt"
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            saved[kind] = str(target)
            logger.info(f"Saved {kind} checkpoint to {target}")
        else:
            logger.warning(f"Run produced no {kind}.pt at {source}")
    return saved


def validate_splits(segmentor: LivestockSegmentor, data_yaml: Path, splits: List[str],
                    outputs_dir: Path) -> Dict[str, Dict[str, float]]:
    """Validate on each requested split and write per-split metric CSVs."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Dict[str, float]] = {}
    with open(data_yaml, "r") as f:
        declared = yaml.safe_load(f) or {}

    for split in splits:
        split_rel = declared.get(split)
        if not split_rel:
            logger.warning(f"data.yaml declares no '{split}' split, skipping validation")
            continue
        split_dir = Path(declared.get("path", data_yaml.parent)) / split_rel
        if not split_dir.exists() or not any(split_dir.iterdir()):
            logger.warning(f"Split '{split}' is empty ({split_dir}), skipping validation")
            continue
        try:
            metrics = segmentor.validate(data_yaml, split=split)
        except Exception as e:  # noqa: BLE001 — surface but don't kill later splits
            logger.error(f"Validation on '{split}' failed: {e}")
            continue
        all_metrics[split] = metrics
        csv_path = outputs_dir / f"metrics_{split}.csv"
        pd.DataFrame([{"split": split, **metrics}]).to_csv(csv_path, index=False)
        logger.info(f"[{split}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    + f" -> {csv_path}")
    return all_metrics


def generate_visualizations(segmentor: LivestockSegmentor, data_yaml: Path,
                            out_dir: Path, limit: int = 8) -> int:
    """
    Predict on sample validation images and export segmentation artifacts
    (binary masks, polygon JSON, overlay visualizations). Returns the number
    of images visualized.
    """
    with open(data_yaml, "r") as f:
        declared = yaml.safe_load(f) or {}
    split_rel = declared.get("val") or declared.get("test") or declared.get("train")
    if not split_rel:
        logger.warning("data.yaml declares no splits; skipping visualizations")
        return 0

    split_dir = Path(declared.get("path", data_yaml.parent)) / split_rel
    images = sorted(p for p in split_dir.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS)[:limit]
    if not images:
        logger.warning(f"No images found under {split_dir}; skipping visualizations")
        return 0

    count = 0
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        try:
            results = segmentor.predict(image, conf=0.1)
        except Exception as e:  # noqa: BLE001 — visualization must not fail the run
            logger.error(f"Prediction failed for {img_path.name}: {e}")
            continue
        if not results:
            continue
        export_segmentation(results, out_dir, img_path.stem, image=image)
        count += 1
    logger.info(f"Wrote segmentation visualizations for {count} images to {out_dir}")
    return count


def write_summary(outputs_dir: Path, summary: Dict[str, Any]) -> Path:
    """Write the run summary JSON and return its path."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary written to {summary_path}")
    return summary_path


def train_segmentor(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full training + validation + visualization flow."""
    config = load_config(args.config)
    paths = config.get("paths", {})
    setup_logging(BASE_DIR / paths.get("logs_dir", "logs"))

    data_yaml = resolve_data_yaml(config, args.data)
    runs_dir = BASE_DIR / paths.get("runs_dir", "runs") / "segment"
    outputs_dir = BASE_DIR / paths.get("outputs_dir", "outputs") / "segmentation"
    device = select_device(args.device or config.get("inference", {}).get("device", "auto"))
    run_name = args.name or DEFAULT_RUN_NAME

    logger.info("=" * 60)
    logger.info("  Livestock Segmentor Training")
    logger.info("=" * 60)
    logger.info(f"Dataset:  {data_yaml}")
    logger.info(f"Device:   {device}")

    if args.resume:
        weights = find_resume_checkpoint(runs_dir, run_name, config)
        logger.info(f"Resuming from {weights}")
    elif args.validate_only and not args.weights:
        weights = None  # let the segmentor pick the fine-tuned checkpoint if present
    else:
        weights = args.weights
        if not weights:
            # Fresh training always starts from pretrained weights (transfer
            # learning), never from a stale fine-tuned checkpoint.
            seg_cfg = config.get("segmentation", {})
            size = args.model_size or seg_cfg.get("model_size", "m")
            weights = f"{resolve_model_name(size, seg_cfg.get('model_family'))}-seg.pt"
    segmentor = LivestockSegmentor(config=config, weights=weights,
                                   model_size=args.model_size, device=device)
    if args.imgsz:
        segmentor.image_size = args.imgsz  # keep val/predict at the training resolution
    logger.info(f"Model:    {segmentor.model_name} (weights: {segmentor.weights_path})")

    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_name": segmentor.model_name,
        "initial_weights": str(segmentor.weights_path),
        "data_yaml": str(data_yaml),
        "device": device,
        "run_name": run_name,
        "resumed": bool(args.resume),
    }

    if not args.validate_only:
        overrides = cli_overrides(args)
        overrides.update({"project": str(runs_dir), "name": run_name,
                          "exist_ok": True, "resume": bool(args.resume)})
        train_kwargs = segmentor.build_train_kwargs(overrides)
        logger.info("Train args: " + ", ".join(f"{k}={v}" for k, v in sorted(train_kwargs.items())))

        try:
            results = segmentor.train(data_yaml, **overrides)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Segmentor training failed: {e}") from e

        run_dir = Path(getattr(results, "save_dir", runs_dir / run_name))
        summary["run_dir"] = str(run_dir)
        summary["checkpoints"] = save_checkpoints(run_dir, config)

        # Keep a copy of the per-epoch metrics next to our outputs.
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            outputs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_csv, outputs_dir / "training_history.csv")

        best = summary["checkpoints"].get("best")
        if best:
            segmentor = LivestockSegmentor(config=config, weights=best, device=device)
            if args.imgsz:
                segmentor.image_size = args.imgsz

    summary["metrics"] = validate_splits(segmentor, data_yaml, args.splits, outputs_dir)
    summary["visualized_images"] = generate_visualizations(
        segmentor, data_yaml, outputs_dir / "visualizations", limit=args.visualize)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(outputs_dir, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the livestock YOLO segmentor (config-driven)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to model_config.yaml")
    parser.add_argument("--data", type=Path, default=None,
                        help="YOLO segmentation dataset descriptor (polygon labels)")
    parser.add_argument("--model-size", choices=list(MODEL_SIZES), default=None,
                        help="Model size override (default from config: segmentation.model_size)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Explicit starting weights (path or Ultralytics name)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="'auto' (default), 'cpu', 'cuda', or CUDA index")
    parser.add_argument("--name", type=str, default=None,
                        help=f"Run name under runs/segment (default: {DEFAULT_RUN_NAME})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the interrupted run from its last checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training; only validate the loaded weights")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Splits to validate after training (default: val test)")
    parser.add_argument("--visualize", type=int, default=8,
                        help="Number of sample images to visualize (default: 8)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        train_segmentor(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        if logger.handlers:
            logger.error(str(e))
        else:
            print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
