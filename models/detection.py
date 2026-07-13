"""
Detection Model for the AI Cattle Analysis System (Phase 3).

Wraps Ultralytics YOLO for livestock detection with:
  - YOLO11 weights when the installed Ultralytics version supports them,
    with automatic fallback to YOLOv8 otherwise
  - Configurable model sizes (n / s / m / l) resolved from
    ``configs/model_config.yaml`` (``detection.model_family`` /
    ``detection.model_size``)
  - Transfer learning from pretrained weights and loading of fine-tuned
    checkpoints (``detection.fine_tuned_weights``)
  - Automatic CPU/CUDA device selection
  - A typed ``predict()`` API returning :class:`Detection` results, plus
    ``train()`` / ``validate()`` used by ``training/train_detector.py``

The legacy module-level functions ``train_yolo()``, ``evaluate_yolo()`` and
``main()`` are preserved for backward compatibility and now run on top of
this implementation.

Usage:
    from models.detection import LivestockDetector

    detector = LivestockDetector()               # config-driven
    detections = detector.predict("cow.jpg")
    metrics = detector.validate("data/processed/data.yaml", split="val")
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"

MODEL_SIZES = ("n", "s", "m", "l")
PREFERRED_FAMILY = "yolo11"
FALLBACK_FAMILY = "yolov8"

# Ultralytics train-argument names keyed by our config keys (detection.training).
_TRAIN_KEY_MAP = {
    "epochs": "epochs",
    "patience": "patience",
    "batch_size": "batch",
    "optimizer": "optimizer",
    "lr": "lr0",
    "weight_decay": "weight_decay",
    "label_smoothing": "label_smoothing",
    "mosaic": "mosaic",
    "mixup": "mixup",
    "copy_paste": "copy_paste",
    "hsv_h": "hsv_h",
    "hsv_s": "hsv_s",
    "hsv_v": "hsv_v",
    "degrees": "degrees",
    "translate": "translate",
    "scale": "scale",
    "perspective": "perspective",
    "multi_scale": "multi_scale",
    "workers": "workers",
    "seed": "seed",
}


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the master YAML configuration (empty dict if the file is missing)."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def yolo11_supported() -> bool:
    """True if the installed Ultralytics package ships YOLO11 model configs."""
    try:
        import ultralytics
    except ImportError:
        return False
    return (Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / "yolo11.yaml").exists()


def resolve_model_name(size: str = "m", family: Optional[str] = None) -> str:
    """
    Resolve a model identifier (e.g. ``yolo11m``) from family + size.

    Falls back from YOLO11 to YOLOv8 when the installed Ultralytics version
    does not support YOLO11.

    Args:
        size: Model size, one of ``n``, ``s``, ``m``, ``l``.
        family: Model family; defaults to YOLO11 when supported.

    Raises:
        ValueError: If ``size`` is not a supported model size.
    """
    size = str(size).lower().strip()
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported model size {size!r}. Choose from {MODEL_SIZES}")
    if family is None:
        family = PREFERRED_FAMILY
    if family == PREFERRED_FAMILY and not yolo11_supported():
        family = FALLBACK_FAMILY
    return f"{family}{size}"


def select_device(device: str = "auto") -> str:
    """Resolve 'auto' to 'cuda' when available, else 'cpu'; pass through otherwise."""
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class Detection:
    """One detected object."""
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[float, float, float, float]  # absolute pixel corners

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box_xyxy": list(self.box_xyxy),
        }


class LivestockDetector:
    """
    Ultralytics YOLO wrapper for livestock detection.

    Weight resolution order:
      1. Explicit ``weights`` argument
      2. Fine-tuned checkpoint from the config (``detection.fine_tuned_weights``)
         when it exists on disk
      3. Pretrained weights for the configured family/size (transfer learning)

    Args:
        config: Parsed ``model_config.yaml`` dict; loaded from disk if omitted.
        weights: Explicit weights path or Ultralytics model name.
        model_size: Override for ``detection.model_size`` (n / s / m / l).
        device: 'auto' (default), 'cpu', 'cuda', or an explicit CUDA index.

    Raises:
        RuntimeError: If the ``ultralytics`` package is not installed.
        FileNotFoundError: If an explicit local weights path does not exist.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weights: Optional[Union[str, Path]] = None,
        model_size: Optional[str] = None,
        device: str = "auto",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "The 'ultralytics' package is required for detection "
                "(pip install ultralytics)"
            ) from e

        self.config = config if config is not None else load_config()
        det_cfg = self.config.get("detection", {})
        self.device = select_device(device or self.config.get("inference", {}).get("device", "auto"))
        self.image_size = int(det_cfg.get("image_size", 640))
        self.conf_threshold = float(det_cfg.get("confidence_threshold", 0.4))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.5))

        size = model_size or det_cfg.get("model_size", "m")
        family = det_cfg.get("model_family")
        self.model_name = resolve_model_name(size, family)
        self.pretrained_weights = f"{self.model_name}.pt"

        self.weights_path, self.is_finetuned = self._resolve_weights(weights, det_cfg)
        self.model = YOLO(str(self.weights_path))

    def _resolve_weights(self, weights: Optional[Union[str, Path]],
                         det_cfg: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply the weight resolution order documented on the class."""
        if weights is not None:
            weights = str(weights)
            # A bare model name (e.g. 'yolo11s.pt') is downloaded by Ultralytics;
            # anything that looks like a path must exist locally.
            if ("/" in weights or "\\" in weights) and not Path(weights).exists():
                raise FileNotFoundError(f"Detector weights not found: {weights}")
            return weights, Path(weights).exists() and not weights.endswith(self.pretrained_weights)

        fine_tuned = det_cfg.get("fine_tuned_weights")
        if fine_tuned:
            fine_tuned_path = BASE_DIR / fine_tuned
            if fine_tuned_path.exists():
                return str(fine_tuned_path), True
        return self.pretrained_weights, False

    @property
    def class_names(self) -> Dict[int, str]:
        """Class id -> name mapping of the loaded model."""
        return dict(self.model.names)

    # ────────────────────────────── inference ──────────────────────────────

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Detection]:
        """
        Run detection on one image (path or RGB/BGR ndarray).

        Returns detections sorted by confidence (highest first).
        """
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold if conf is None else conf,
            iou=self.iou_threshold if iou is None else iou,
            max_det=max_det or int(self.config.get("inference", {}).get("max_detections", 10)),
            imgsz=self.image_size,
            device=self.device,
            verbose=verbose,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=float(box.conf.item()),
                    box_xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                ))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    # ─────────────────────────── training / validation ───────────────────────────

    def build_train_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Translate ``detection.training`` config keys into Ultralytics train
        arguments; ``overrides`` (already in Ultralytics naming) win.
        """
        train_cfg = self.config.get("detection", {}).get("training", {})
        kwargs: Dict[str, Any] = {
            "imgsz": self.image_size,
            "device": self.device,
        }
        for config_key, ultra_key in _TRAIN_KEY_MAP.items():
            if config_key in train_cfg:
                kwargs[ultra_key] = train_cfg[config_key]
        if overrides:
            kwargs.update(overrides)
        return kwargs

    def train(self, data: Union[str, Path], **overrides: Any) -> Any:
        """
        Fine-tune on a YOLO dataset descriptor (``data.yaml``).

        All hyperparameters come from the config; keyword arguments override
        them using Ultralytics naming (``epochs``, ``batch``, ``lr0``, ...).
        Returns the Ultralytics results object.

        Raises:
            FileNotFoundError: If ``data`` does not exist.
        """
        data = Path(data)
        if not data.exists():
            raise FileNotFoundError(
                f"Dataset descriptor not found: {data} "
                f"(run dataset/download_datasets.py and dataset/unify_datasets.py first)"
            )
        kwargs = self.build_train_kwargs(overrides)
        return self.model.train(data=str(data), **kwargs)

    def validate(self, data: Union[str, Path], split: str = "val",
                 **kwargs: Any) -> Dict[str, float]:
        """
        Evaluate on a dataset split and return the standard box metrics.

        Returns:
            Dict with mAP50, mAP50-95, precision, recall, and fitness.
        """
        data = Path(data)
        if not data.exists():
            raise FileNotFoundError(f"Dataset descriptor not found: {data}")
        metrics = self.model.val(data=str(data), split=split, imgsz=self.image_size,
                                 device=self.device, **kwargs)
        return extract_box_metrics(metrics)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the current model weights to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path


def extract_box_metrics(metrics: Any) -> Dict[str, float]:
    """Extract the standard box metrics from an Ultralytics validation result."""
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness),
    }


# ──────────────────────── legacy API (backward compatible) ────────────────────────


def train_yolo() -> Any:
    """
    Deprecated: use ``training/train_detector.py``.

    Preserved legacy entry point — trains on ``data/livestock_yolo/data.yaml``
    and saves the best weights to ``checkpoints/yolo_best.pt`` as before.
    Returns the underlying Ultralytics YOLO model.
    """
    data_yaml = BASE_DIR / "data" / "livestock_yolo" / "data.yaml"
    checkpoints_dir = BASE_DIR / "checkpoints"
    runs_dir = BASE_DIR / "runs" / "detect"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    detector = LivestockDetector(weights="yolov8m.pt")
    print(f"Starting training on {data_yaml}...")
    detector.train(data_yaml, epochs=30, patience=10, imgsz=640,
                   project=str(runs_dir), name="train_livestock", exist_ok=True)

    best_weights = runs_dir / "train_livestock" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, checkpoints_dir / "yolo_best.pt")
        print(f"Saved best weights to {checkpoints_dir / 'yolo_best.pt'}")
    return detector.model


def evaluate_yolo(model: Any) -> None:
    """
    Deprecated: use ``training/train_detector.py --validate-only``.

    Preserved legacy entry point — evaluates on the test split and writes
    ``outputs/detection_metrics.csv`` as before.
    """
    import pandas as pd

    data_yaml = BASE_DIR / "data" / "livestock_yolo" / "data.yaml"
    runs_dir = BASE_DIR / "runs" / "detect"
    outputs_dir = BASE_DIR / "outputs" / "detection"
    metrics_csv = BASE_DIR / "outputs" / "detection_metrics.csv"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating YOLO on test set...")
    metrics = model.val(data=str(data_yaml), split="test",
                        project=str(runs_dir), name="val_livestock", exist_ok=True)
    box = extract_box_metrics(metrics)
    df = pd.DataFrame({
        "Metric": ["mAP50", "mAP50-95", "Precision", "Recall", "Fitness"],
        "Value": [box["mAP50"], box["mAP50-95"], box["precision"],
                  box["recall"], box["fitness"]],
    })
    df.to_csv(metrics_csv, index=False)
    print(f"Saved detection metrics to {metrics_csv}")

    test_images_dir = BASE_DIR / "data" / "processed" / "images" / "test"
    if test_images_dir.exists() and list(test_images_dir.glob("*.jpg")):
        print("Generating detection outputs for test set...")
        model.predict(source=str(test_images_dir), save=True,
                      project=str(outputs_dir.parent), name="detection", exist_ok=True)


def main() -> None:
    """Deprecated legacy pipeline: train then evaluate (kept for compatibility)."""
    model = train_yolo()
    evaluate_yolo(model)
    print("YOLO detection pipeline complete.")


if __name__ == "__main__":
    main()
