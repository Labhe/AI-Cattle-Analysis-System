"""
Species Classifier Training for the AI Cattle Analysis System (Phase 5).

Trains the species classifier (EfficientNetV2-S / ConvNeXt-Tiny / Swin-T,
configurable via ``species_classifier.model_name``) on the image-folder
classification split produced by ``dataset/unify_datasets.py``
(``data/processed/classification/{train,val,test}/<label>/``). Folder labels
are mapped onto the configured species classes; unrecognized folders (e.g.
breed folders) are skipped with a warning.

All hyperparameters come from ``configs/model_config.yaml``
(``species_classifier`` section): learning rate, optimizer, scheduler,
epochs, batch size, image size, dropout, and label smoothing — with CLI
overrides for experiments.

Features: transfer learning, automatic CPU/CUDA, mixed precision on CUDA,
early stopping, best/last checkpoints, resume, and optional MixUp
(reusing ``dataset/augmentation.py``).

Outputs:
  - checkpoints/species_classifier_best.pth   best weights (used by inference.py)
  - checkpoints/species_classifier_last.pth   full training state (resume source)
  - outputs/species_classifier/training_history.csv
  - outputs/species_classifier/metrics_<split>.csv
  - outputs/species_classifier/confusion_matrix_<split>.png (+ .csv)
  - outputs/species_classifier/classification_report_<split>.txt
  - outputs/species_classifier/training_summary.json
  - logs/train_species_classifier.log

Usage:
    python -m training.train_species_classifier
    python -m training.train_species_classifier --arch convnext_tiny --epochs 30
    python -m training.train_species_classifier --resume
    python -m training.train_species_classifier --validate-only
"""

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset.augmentation import get_classification_transforms, mixup_batch
from models.detection import load_config, select_device
from models.species_classifier import (
    SUPPORTED_ARCHITECTURES,
    build_species_classifier,
    load_species_classes,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder-name aliases mapped onto the configured class names (lowercase keys).
SPECIES_FOLDER_ALIASES = {
    "cattle": "Cow", "cow": "Cow", "cows": "Cow",
    "bull": "Bull", "bulls": "Bull",
    "buffalo": "Buffalo", "buffaloes": "Buffalo", "water buffalo": "Buffalo",
    "yak": "Yak", "ox": "Ox", "oxen": "Ox",
    "goat": "Goat", "goats": "Goat",
    "sheep": "Sheep", "lamb": "Sheep",
    "horse": "Horse", "horses": "Horse",
    "camel": "Camel", "camels": "Camel",
    "dog": "Dog", "dogs": "Dog",
    "cat": "Cat", "cats": "Cat",
    "human": "Human", "person": "Human",
}

logger = logging.getLogger("training.species_classifier")


def setup_logging(logs_dir: Path) -> None:
    """Configure console + file logging (idempotent)."""
    if logger.handlers:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / "train_species_classifier.log")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class SpeciesImageFolder(Dataset):
    """
    Image-folder dataset mapping directory names onto the configured species.

    Directory names are matched case-insensitively against the class names
    and :data:`SPECIES_FOLDER_ALIASES`; non-matching folders are skipped.
    Unreadable images are skipped gracefully at access time.
    """

    def __init__(self, root: Path, split: str, class_names: List[str], transform: Any):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.skipped_folders: List[str] = []

        name_to_idx = {name.lower(): i for i, name in enumerate(class_names)}
        alias_to_idx = {
            alias: name_to_idx[target.lower()]
            for alias, target in SPECIES_FOLDER_ALIASES.items()
            if target.lower() in name_to_idx
        }
        alias_to_idx.update(name_to_idx)

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Classification split directory not found: {split_dir} "
                f"(run dataset/unify_datasets.py first)"
            )
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_idx = alias_to_idx.get(class_dir.name.strip().lower())
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


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build the configured optimizer (adamw / adam / sgd)."""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}. Choose from: adamw, adam, sgd")


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int,
                    min_lr: float) -> Optional[Any]:
    """Build the configured LR scheduler (cosine / step / plateau / none)."""
    name = (name or "none").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          factor=0.5, patience=3, min_lr=min_lr)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler {name!r}. Choose from: cosine, step, plateau, none")


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: str, optimizer: Optional[torch.optim.Optimizer] = None,
              scaler: Optional[torch.amp.GradScaler] = None,
              mixup_alpha: float = 0.0) -> Tuple[float, float]:
    """One train (optimizer given) or eval epoch; returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mixed = training and mixup_alpha > 0
            if mixed:
                images, targets_a, targets_b, lam = mixup_batch(images, labels, mixup_alpha)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                if mixed:
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader,
                        device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; returns (y_true, y_pred)."""
    model.eval()
    trues, preds = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(labels.numpy())
    if not trues:
        return np.array([]), np.array([])
    return np.concatenate(trues), np.concatenate(preds)


def evaluate_split(model: nn.Module, dataset: SpeciesImageFolder, split: str,
                   class_names: List[str], device: str, batch_size: int,
                   workers: int, outputs_dir: Path) -> Dict[str, float]:
    """
    Full evaluation on a split: metrics CSV, confusion matrix (PNG + CSV),
    and classification report (TXT). Returns the headline metrics.
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

    plt.figure(figsize=(max(8, len(class_names) * 0.8),) * 2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Species Confusion Matrix ({split})")
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


def train_species_classifier(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full training + evaluation flow; returns the summary dict."""
    config = load_config(args.config)
    paths = config.get("paths", {})
    setup_logging(BASE_DIR / paths.get("logs_dir", "logs"))

    clf_cfg = config.get("species_classifier", {})
    train_cfg = clf_cfg.get("training", {})
    device = select_device(args.device or config.get("inference", {}).get("device", "auto"))
    class_names = load_species_classes(config)

    architecture = args.arch or clf_cfg.get("model_name", "efficientnet_v2_s")
    image_size = args.imgsz or int(clf_cfg.get("image_size", 224))
    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    batch_size = args.batch or int(train_cfg.get("batch_size", 32))
    lr = args.lr or float(train_cfg.get("lr", 1e-3))
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    dropout = float(train_cfg.get("dropout", 0.3))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    patience = args.patience or int(train_cfg.get("patience", 10))
    workers = args.workers if args.workers is not None else int(train_cfg.get("workers", 4))
    use_amp = bool(train_cfg.get("mixed_precision", True)) and device == "cuda"
    torch.manual_seed(int(train_cfg.get("seed", 42)))

    data_root = args.data or BASE_DIR / paths.get("processed_dir", "data/processed") / "classification"
    best_path = BASE_DIR / clf_cfg.get("weights", "checkpoints/species_classifier_best.pth")
    last_path = BASE_DIR / clf_cfg.get("last_checkpoint", "checkpoints/species_classifier_last.pth")
    outputs_dir = BASE_DIR / paths.get("outputs_dir", "outputs") / "species_classifier"

    logger.info("=" * 60)
    logger.info("  Species Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Architecture: {architecture} | Device: {device} | AMP: {use_amp}")
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Dataset root: {data_root}")

    model = build_species_classifier(architecture, num_classes=len(class_names),
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
        train_set = SpeciesImageFolder(Path(data_root), "train", class_names,
                                       get_classification_transforms("train", image_size))
        val_set = SpeciesImageFolder(Path(data_root), "val", class_names,
                                     get_classification_transforms("val", image_size))
        for ds, name in ((train_set, "train"), (val_set, "val")):
            if ds.skipped_folders:
                logger.warning(f"[{name}] skipped non-species folders: {ds.skipped_folders}")
        if len(train_set) == 0 or len(val_set) == 0:
            raise RuntimeError(
                f"No species samples found under {data_root} "
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
                        "checkpoints": {"best": str(best_path), "last": str(last_path)}})

        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    else:
        weights = Path(args.weights) if args.weights else best_path
        if not weights.exists():
            raise FileNotFoundError(f"No weights to validate: {weights}")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        logger.info(f"Loaded weights for validation: {weights}")

    # Final evaluation: metrics, confusion matrix, classification report.
    summary["metrics"] = {}
    for split in args.splits:
        try:
            dataset = SpeciesImageFolder(Path(data_root), split, class_names,
                                         get_classification_transforms("val", image_size))
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue
        if len(dataset) == 0:
            logger.warning(f"Split '{split}' has no species samples; skipping")
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
        description="Train the species classifier (config-driven)")
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
        train_species_classifier(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        if logger.handlers:
            logger.error(str(e))
        else:
            print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
