"""
Breed Classifier Training for the AI Cattle Analysis System (Phase 6).

Trains the cattle breed classifier (EfficientNetV2-S / ConvNeXt-Tiny / Swin-T,
configurable via ``breed_classifier.model_name``) on the image-folder
classification split produced by ``dataset/unify_datasets.py``
(``data/processed/classification/{train,val,test}/<breed>/``). Species-level
folders (cow, goat, ...) are skipped — they belong to the species classifier;
the remaining folder names become the breed classes. The trained class list
is persisted to ``checkpoints/breed_classes.json`` so inference always uses
the exact classes the checkpoint was trained on.

All hyperparameters come from ``configs/model_config.yaml``
(``breed_classifier`` section): architecture, learning rate, optimizer,
scheduler, epochs, batch size, image size, dropout, and label smoothing —
with CLI overrides for experiments. Generic training utilities (optimizer /
scheduler factories, epoch loop, prediction collection) are reused from
``training/train_species_classifier.py``.

Features: transfer learning, automatic CPU/CUDA, mixed precision on CUDA,
early stopping, best/last checkpoints, resume, and optional MixUp.

Outputs:
  - checkpoints/breed_classifier_best.pth    best weights (used by inference.py)
  - checkpoints/breed_classifier_last.pth    full training state (resume source)
  - checkpoints/breed_classes.json           trained breed class list
  - outputs/breed_classifier/training_history.csv
  - outputs/breed_classifier/metrics_<split>.csv
  - outputs/breed_classifier/confusion_matrix_<split>.png (+ .csv)
  - outputs/breed_classifier/per_class_accuracy_<split>.csv
  - outputs/breed_classifier/classification_report_<split>.txt
  - outputs/breed_classifier/training_summary.json
  - logs/train_breed_classifier.log

Usage:
    python -m training.train_breed_classifier
    python -m training.train_breed_classifier --arch efficientnet_v2_s --epochs 40
    python -m training.train_breed_classifier --resume
    python -m training.train_breed_classifier --validate-only
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset.augmentation import get_classification_transforms
from models.breed_classifier import (
    SUPPORTED_ARCHITECTURES,
    build_breed_classifier,
)
from models.detection import load_config, select_device
from training.train_species_classifier import (  # reuse generic training utilities
    SPECIES_FOLDER_ALIASES,
    build_optimizer,
    build_scheduler,
    collect_predictions,
    run_epoch,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger("training.breed_classifier")


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "train_breed_classifier.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def is_species_folder(name: str) -> bool:
    """True for folder names that denote a species rather than a breed."""
    return name.strip().lower() in SPECIES_FOLDER_ALIASES


def discover_breed_classes(root: Path, declared: Optional[List[str]]) -> List[str]:
    """
    Resolve the breed class list: the config-declared list if present,
    otherwise the sorted breed folder names of the train split.
    """
    if declared:
        return [str(c) for c in declared]
    train_dir = root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Classification split directory not found: {train_dir} "
            f"(run dataset/unify_datasets.py first)"
        )
    classes = sorted(
        d.name for d in train_dir.iterdir()
        if d.is_dir() and not is_species_folder(d.name)
        and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())
    )
    if not classes:
        raise RuntimeError(
            f"No breed folders found under {train_dir} — species-level folders "
            f"are excluded; provide breed image folders or declare "
            f"breed_classifier.classes in the config"
        )
    return classes


class BreedImageFolder(Dataset):
    """
    Image-folder dataset over breed directories.

    Species-level folders are skipped (they belong to the species classifier);
    folders not in the trained class list are skipped with a warning.
    Unreadable images are skipped gracefully at access time.
    """

    def __init__(self, root: Path, split: str, class_names: List[str], transform: Any):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.skipped_folders: List[str] = []

        name_to_idx = {name: i for i, name in enumerate(class_names)}
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Classification split directory not found: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if is_species_folder(class_dir.name):
                continue  # species folders are expected; skip silently
            class_idx = name_to_idx.get(class_dir.name)
            if class_idx is None:
                self.skipped_folders.append(class_dir.name)
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Skip unreadable files by advancing to the next sample (bounded scan).
        for attempt in range(len(self.samples)):
            path, label = self.samples[(idx + attempt) % len(self.samples)]
            image = cv2.imread(str(path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = self.transform(image=image)["image"]
                return tensor, label
            logger.warning(f"Skipping unreadable image: {path}")
        raise RuntimeError("No readable images in the dataset")

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0] * num_classes
        for _, label in self.samples:
            counts[label] += 1
        return counts


def evaluate_split(model: nn.Module, dataset: BreedImageFolder, split: str,
                   class_names: List[str], device: str, batch_size: int,
                   workers: int, outputs_dir: Path) -> Dict[str, float]:
    """
    Full evaluation on a split: metrics CSV, confusion matrix (CSV + PNG),
    per-class accuracy CSV, and classification report (TXT).
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    y_true, y_pred = collect_predictions(model, loader, device)
    if y_true.size == 0:
        logger.warning(f"Split '{split}' has no samples; skipping evaluation")
        return {}

    labels = list(range(len(class_names)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "samples": int(y_true.size),
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"split": split, **metrics}]).to_csv(
        outputs_dir / f"metrics_{split}.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        outputs_dir / f"confusion_matrix_{split}.csv")

    # Per-class accuracy (recall of each true class)
    support = cm.sum(axis=1)
    per_class = np.divide(cm.diagonal(), support, where=support > 0,
                          out=np.zeros(len(class_names), dtype=float))
    pd.DataFrame({
        "breed": class_names,
        "support": support,
        "accuracy": np.round(per_class, 4),
    }).to_csv(outputs_dir / f"per_class_accuracy_{split}.csv", index=False)

    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=class_names, zero_division=0)
    (outputs_dir / f"classification_report_{split}.txt").write_text(report)

    try:
        _plot_confusion_matrix(cm, class_names,
                               outputs_dir / f"confusion_matrix_{split}.png", split)
    except Exception as e:  # noqa: BLE001 — plotting must not fail the run
        logger.warning(f"Could not render confusion matrix plot: {e}")

    logger.info(f"[{split}] accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f} ({metrics['samples']} samples)")
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                           out_path: Path, split: str) -> None:
    """Render the confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(max(8, len(class_names) * 0.6),) * 2)
    sns.heatmap(cm, annot=len(class_names) <= 25, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Breed Confusion Matrix ({split})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_last_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                         scheduler: Optional[Any], scaler: torch.amp.GradScaler,
                         epoch: int, best_acc: float, epochs_no_improve: int,
                         history: List[Dict[str, float]], architecture: str,
                         class_names: List[str]) -> None:
    """Persist the full training state for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "architecture": architecture,
        "class_names": class_names,
    }, path)


def train_breed_classifier(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full training + evaluation flow; returns the summary dict."""
    config = load_config(args.config)
    paths = config.get("paths", {})
    setup_logging(BASE_DIR / paths.get("logs_dir", "logs"))

    clf_cfg = config.get("breed_classifier", {})
    train_cfg = clf_cfg.get("training", {})
    device = select_device(args.device or config.get("inference", {}).get("device", "auto"))

    architecture = args.arch or clf_cfg.get("model_name", "convnext_tiny")
    image_size = args.imgsz or int(clf_cfg.get("image_size", 224))
    epochs = args.epochs or int(train_cfg.get("epochs", 60))
    batch_size = args.batch or int(train_cfg.get("batch_size", 32))
    lr = args.lr or float(train_cfg.get("lr", 5e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 5e-5))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    dropout = float(train_cfg.get("dropout", 0.4))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    patience = args.patience or int(train_cfg.get("patience", 10))
    workers = args.workers if args.workers is not None else int(train_cfg.get("workers", 4))
    use_amp = bool(train_cfg.get("mixed_precision", True)) and device == "cuda"
    torch.manual_seed(int(train_cfg.get("seed", 42)))

    data_root = Path(args.data) if args.data \
        else BASE_DIR / paths.get("processed_dir", "data/processed") / "classification"
    best_path = BASE_DIR / clf_cfg.get("weights", "checkpoints/breed_classifier_best.pth")
    last_path = BASE_DIR / clf_cfg.get("last_checkpoint", "checkpoints/breed_classifier_last.pth")
    classes_path = BASE_DIR / clf_cfg.get("classes_file", "checkpoints/breed_classes.json")
    outputs_dir = BASE_DIR / paths.get("outputs_dir", "outputs") / "breed_classifier"

    class_names = discover_breed_classes(data_root, clf_cfg.get("classes"))

    logger.info("=" * 60)
    logger.info("  Breed Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Architecture: {architecture} | Device: {device} | AMP: {use_amp}")
    logger.info(f"Breed classes ({len(class_names)}): {class_names}")
    logger.info(f"Dataset root: {data_root}")

    model = build_breed_classifier(architecture, num_classes=len(class_names),
                                   dropout=dropout, pretrained=not args.no_pretrained)
    model.to(device)

    summary: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture,
        "device": device,
        "mixed_precision": use_amp,
        "image_size": image_size,
        "classes": class_names,
    }

    if not args.validate_only:
        train_set = BreedImageFolder(data_root, "train", class_names,
                                     get_classification_transforms("train", image_size))
        val_set = BreedImageFolder(data_root, "val", class_names,
                                   get_classification_transforms("val", image_size))
        for ds, name in ((train_set, "train"), (val_set, "val")):
            if ds.skipped_folders:
                logger.warning(f"[{name}] skipped unknown breed folders: {ds.skipped_folders}")
        if len(train_set) == 0 or len(val_set) == 0:
            raise RuntimeError(
                f"No breed samples found under {data_root} "
                f"(train={len(train_set)}, val={len(val_set)})"
            )
        logger.info(f"Samples: train={len(train_set)}, val={len(val_set)}")
        logger.info("Train class counts: " + ", ".join(
            f"{n}={c}" for n, c in zip(class_names, train_set.class_counts(len(class_names))) if c))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=workers, pin_memory=device == "cuda")
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=workers, pin_memory=device == "cuda")

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = build_optimizer(args.optimizer or train_cfg.get("optimizer", "adamw"),
                                    model, lr, weight_decay)
        scheduler = build_scheduler(args.scheduler or train_cfg.get("scheduler", "cosine"),
                                    optimizer, epochs, min_lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch, best_acc, epochs_no_improve = 0, 0.0, 0
        history: List[Dict[str, float]] = []

        if args.resume:
            if not last_path.exists():
                raise FileNotFoundError(f"No checkpoint to resume from: {last_path}")
            state = torch.load(last_path, map_location=device, weights_only=False)
            if state.get("architecture") != architecture:
                raise RuntimeError(
                    f"Resume checkpoint was trained with {state.get('architecture')!r}, "
                    f"but current architecture is {architecture!r}"
                )
            if state.get("class_names") != class_names:
                raise RuntimeError(
                    "Resume checkpoint was trained with a different breed class list; "
                    "delete the checkpoint or restore the original dataset"
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            best_acc = state["best_acc"]
            epochs_no_improve = state.get("epochs_no_improve", 0)
            history = state.get("history", [])
            logger.info(f"Resumed from epoch {start_epoch} (best val acc {best_acc:.4f})")

        # Persist the trained class list + architecture for inference
        # (BreedPredictor builds the exact backbone the checkpoint used).
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(classes_path, "w") as f:
            json.dump({"classes": class_names, "architecture": architecture}, f, indent=2)

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                              optimizer=optimizer, scaler=scaler,
                                              mixup_alpha=mixup_alpha)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            elif scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "train_acc": train_acc, "val_loss": val_loss,
                            "val_acc": val_acc, "lr": current_lr})
            logger.info(f"Epoch {epoch + 1}/{epochs} | "
                        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                        f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {current_lr:.2e}")

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best val acc {best_acc:.4f} — saved {best_path}")
            else:
                epochs_no_improve += 1

            save_last_checkpoint(last_path, model, optimizer, scheduler, scaler,
                                 epoch, best_acc, epochs_no_improve, history,
                                 architecture, class_names)

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break

        outputs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(outputs_dir / "training_history.csv", index=False)
        summary.update({"epochs_run": len(history), "best_val_acc": best_acc,
                        "checkpoints": {"best": str(best_path), "last": str(last_path),
                                        "classes": str(classes_path)}})

        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    else:
        weights = Path(args.weights) if args.weights else best_path
        if not weights.exists():
            raise FileNotFoundError(f"No weights to validate: {weights}")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        logger.info(f"Loaded weights for validation: {weights}")

    # Final evaluation: metrics, confusion matrix, per-class accuracy, report.
    summary["metrics"] = {}
    for split in args.splits:
        try:
            dataset = BreedImageFolder(data_root, split, class_names,
                                       get_classification_transforms("val", image_size))
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue
        if len(dataset) == 0:
            logger.warning(f"Split '{split}' has no breed samples; skipping")
            continue
        summary["metrics"][split] = evaluate_split(
            model, dataset, split, class_names, device, batch_size, workers, outputs_dir)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary written to {outputs_dir / 'training_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the breed classifier (config-driven)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to model_config.yaml")
    parser.add_argument("--data", type=Path, default=None,
                        help="Classification dataset root (default: data/processed/classification)")
    parser.add_argument("--arch", choices=list(SUPPORTED_ARCHITECTURES), default=None,
                        help="Architecture override (default from config: model_name)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Weights for --validate-only (default: best checkpoint)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default=None)
    parser.add_argument("--scheduler", choices=["cosine", "step", "plateau", "none"], default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="'auto' (default), 'cpu', 'cuda', or CUDA index")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch instead of ImageNet weights")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training; only evaluate the loaded weights")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Splits to evaluate after training (default: val test)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        train_breed_classifier(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        if logger.handlers:
            logger.error(str(e))
        else:
            print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
