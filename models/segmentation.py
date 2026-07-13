"""
Segmentation Models for the AI Cattle Analysis System (Phase 4).

Primary model: Ultralytics YOLO11 instance segmentation (automatic fallback
to YOLOv8-seg when YOLO11 is unsupported), wrapped by
:class:`LivestockSegmentor` with:

  - Transfer learning from pretrained ``*-seg`` weights and loading of
    fine-tuned checkpoints (``segmentation.fine_tuned_weights``)
  - Configurable model sizes (n / s / m / l), image size, and confidence /
    IoU thresholds — all read from ``configs/model_config.yaml``
  - Automatic CPU/CUDA selection (shared with the Phase 3 detector)
  - A typed ``predict()`` API returning :class:`SegmentationResult` objects
    that carry the polygon mask, the rasterized binary mask, and the
    per-instance confidence
  - ``predict_parts()`` isolating the complete animal plus head, torso,
    legs, and tail: body-part classes are used directly when the loaded
    checkpoint was trained on them (``segmentation.classes``); otherwise
    parts are derived from the whole-animal mask with a documented
    geometric estimation (:func:`estimate_body_parts`)
  - Mask export (binary PNGs + polygon JSON) and overlay visualization

The legacy U-Net (``UNetResNet34``) is preserved unchanged as the fallback
segmentor used by ``inference.py``, together with its ``DiceBCE`` loss,
metric helpers, and training entry points.

Usage:
    from models.segmentation import LivestockSegmentor

    segmentor = LivestockSegmentor()                # config-driven
    results = segmentor.predict("cow.jpg")
    parts = segmentor.predict_parts("cow.jpg")      # animal/head/torso/legs/tail
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.detection import (  # noqa: E402 — reuse Phase 3 utilities
    load_config,
    resolve_model_name,
    select_device,
)

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs" / "segmentation"
METRICS_CSV = BASE_DIR / "outputs" / "segmentation_metrics.csv"

BODY_PART_NAMES = ("animal", "head", "torso", "legs", "tail")
DEFAULT_SEG_CLASSES = {0: "animal", 1: "head", 2: "torso", 3: "legs", 4: "tail"}

# BGR-independent RGB colors used by overlay visualizations.
PART_COLORS: Dict[str, Tuple[int, int, int]] = {
    "animal": (0, 220, 120),
    "head": (255, 140, 0),
    "torso": (0, 150, 255),
    "legs": (200, 0, 200),
    "tail": (255, 220, 0),
}
_FALLBACK_COLOR = (0, 220, 120)

# Ultralytics train-argument names keyed by our config keys (segmentation.training).
_SEG_TRAIN_KEY_MAP = {
    "epochs": "epochs",
    "patience": "patience",
    "batch_size": "batch",
    "optimizer": "optimizer",
    "lr": "lr0",
    "weight_decay": "weight_decay",
    "workers": "workers",
    "seed": "seed",
}


@dataclass
class SegmentationResult:
    """One segmented instance."""
    class_id: int
    class_name: str
    confidence: float
    binary_mask: np.ndarray = field(repr=False)  # (H, W) uint8 in {0, 255}
    polygon: np.ndarray = field(repr=False)      # (N, 2) float32 absolute pixel coords
    box_xyxy: Tuple[float, float, float, float]
    source: str = "model"                        # 'model' | 'estimated'

    @property
    def area(self) -> float:
        """Mask area in pixels."""
        return float(np.count_nonzero(self.binary_mask))

    def to_dict(self, include_polygon: bool = True) -> Dict[str, Any]:
        """JSON-serializable summary (the binary mask is intentionally omitted)."""
        payload: Dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box_xyxy": list(self.box_xyxy),
            "area_px": self.area,
            "source": self.source,
        }
        if include_polygon:
            payload["polygon"] = self.polygon.astype(float).tolist()
        return payload


class LivestockSegmentor:
    """
    Ultralytics YOLO segmentation wrapper for livestock analysis.

    Weight resolution order (mirrors the Phase 3 detector):
      1. Explicit ``weights`` argument
      2. Fine-tuned checkpoint (``segmentation.fine_tuned_weights``) when it
         exists on disk
      3. Pretrained ``*-seg`` weights for the configured family/size

    Args:
        config: Parsed ``model_config.yaml`` dict; loaded from disk if omitted.
        weights: Explicit weights path or Ultralytics model name.
        model_size: Override for ``segmentation.model_size`` (n / s / m / l).
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
                "The 'ultralytics' package is required for segmentation "
                "(pip install ultralytics)"
            ) from e

        self.config = config if config is not None else load_config()
        seg_cfg = self.config.get("segmentation", {})
        self.device = select_device(device or self.config.get("inference", {}).get("device", "auto"))
        self.image_size = int(seg_cfg.get("image_size", 640))
        self.conf_threshold = float(seg_cfg.get("confidence_threshold", 0.4))
        self.iou_threshold = float(seg_cfg.get("iou_threshold", 0.5))
        self.part_classes = {
            int(k): str(v).lower()
            for k, v in seg_cfg.get("classes", DEFAULT_SEG_CLASSES).items()
        }

        size = model_size or seg_cfg.get("model_size", "m")
        family = seg_cfg.get("model_family")
        self.model_name = f"{resolve_model_name(size, family)}-seg"
        self.pretrained_weights = f"{self.model_name}.pt"

        self.weights_path, self.is_finetuned = self._resolve_weights(weights, seg_cfg)
        self.model = YOLO(str(self.weights_path))

    def _resolve_weights(self, weights: Optional[Union[str, Path]],
                         seg_cfg: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply the weight resolution order documented on the class."""
        if weights is not None:
            weights = str(weights)
            if ("/" in weights or "\\" in weights) and not Path(weights).exists():
                raise FileNotFoundError(f"Segmentor weights not found: {weights}")
            return weights, Path(weights).exists() and not weights.endswith(self.pretrained_weights)

        fine_tuned = seg_cfg.get("fine_tuned_weights")
        if fine_tuned:
            fine_tuned_path = BASE_DIR / fine_tuned
            if fine_tuned_path.exists():
                return str(fine_tuned_path), True
        return self.pretrained_weights, False

    @property
    def class_names(self) -> Dict[int, str]:
        """Class id -> name mapping of the loaded model."""
        return dict(self.model.names)

    def is_part_model(self) -> bool:
        """True when the loaded checkpoint was trained with body-part classes."""
        names = {str(n).lower() for n in self.class_names.values()}
        return {"head", "torso", "legs"}.issubset(names)

    # ────────────────────────────── inference ──────────────────────────────

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        verbose: bool = False,
    ) -> List[SegmentationResult]:
        """
        Segment one image (path or RGB/BGR ndarray).

        Each returned :class:`SegmentationResult` carries the polygon in
        absolute pixel coordinates, a binary mask rasterized at the original
        image size, and the instance confidence. Results are sorted by
        confidence (highest first).
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
        segmentations: List[SegmentationResult] = []
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            height, width = result.orig_shape
            for polygon, box in zip(result.masks.xy, result.boxes):
                if polygon is None or len(polygon) < 3:
                    continue
                polygon = np.asarray(polygon, dtype=np.float32)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
                class_id = int(box.cls.item())
                segmentations.append(SegmentationResult(
                    class_id=class_id,
                    class_name=str(self.class_names.get(class_id, class_id)).lower(),
                    confidence=float(box.conf.item()),
                    binary_mask=mask,
                    polygon=polygon,
                    box_xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                ))
        return sorted(segmentations, key=lambda s: s.confidence, reverse=True)

    def predict_parts(
        self,
        source: Optional[Union[str, Path, np.ndarray]] = None,
        results: Optional[List[SegmentationResult]] = None,
        conf: Optional[float] = None,
    ) -> Dict[str, Optional[SegmentationResult]]:
        """
        Isolate the complete animal and its head / torso / legs / tail.

        When the loaded checkpoint was trained with body-part classes, the
        model's own masks are returned (highest confidence per part). For
        whole-animal models, parts are derived from the best animal mask via
        :func:`estimate_body_parts` and flagged ``source='estimated'``.

        Args:
            source: Image to segment (ignored when ``results`` is given).
            results: Precomputed output of :meth:`predict` to reuse.
            conf: Confidence threshold override for the internal predict call.

        Returns:
            Dict keyed by :data:`BODY_PART_NAMES`; missing parts map to None.
        """
        if results is None:
            if source is None:
                raise ValueError("predict_parts requires either 'source' or 'results'")
            results = self.predict(source, conf=conf)

        parts: Dict[str, Optional[SegmentationResult]] = {n: None for n in BODY_PART_NAMES}
        if not results:
            return parts

        if self.is_part_model():
            for res in results:  # already confidence-sorted
                if res.class_name in parts and parts[res.class_name] is None:
                    parts[res.class_name] = res
            if parts["animal"] is None:
                found = [r for r in parts.values() if r is not None]
                if found:
                    parts["animal"] = _union_results(found)
            return parts

        whole = results[0]
        parts["animal"] = whole
        for name, mask in estimate_body_parts(whole.binary_mask).items():
            polygon = mask_to_polygon(mask)
            if polygon is None:
                continue
            ys, xs = np.nonzero(mask)
            parts[name] = SegmentationResult(
                class_id=-1,
                class_name=name,
                confidence=whole.confidence,
                binary_mask=mask,
                polygon=polygon,
                box_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                source="estimated",
            )
        return parts

    # ─────────────────────────── training / validation ───────────────────────────

    def build_train_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Translate ``segmentation.training`` config keys into Ultralytics train
        arguments; ``overrides`` (already in Ultralytics naming) win.
        """
        train_cfg = self.config.get("segmentation", {}).get("training", {})
        kwargs: Dict[str, Any] = {"imgsz": self.image_size, "device": self.device}
        for config_key, ultra_key in _SEG_TRAIN_KEY_MAP.items():
            if config_key in train_cfg:
                kwargs[ultra_key] = train_cfg[config_key]
        if overrides:
            kwargs.update(overrides)
        return kwargs

    def train(self, data: Union[str, Path], **overrides: Any) -> Any:
        """
        Fine-tune on a YOLO segmentation dataset descriptor (``data.yaml``
        with polygon labels).

        All hyperparameters come from the config; keyword arguments override
        them using Ultralytics naming. Returns the Ultralytics results object.

        Raises:
            FileNotFoundError: If ``data`` does not exist.
        """
        data = Path(data)
        if not data.exists():
            raise FileNotFoundError(
                f"Dataset descriptor not found: {data} "
                f"(a YOLO segmentation dataset with polygon labels is required)"
            )
        return self.model.train(data=str(data), **self.build_train_kwargs(overrides))

    def validate(self, data: Union[str, Path], split: str = "val",
                 **kwargs: Any) -> Dict[str, float]:
        """
        Evaluate on a dataset split.

        Returns:
            Dict with box and mask mAP / precision / recall plus fitness.
        """
        data = Path(data)
        if not data.exists():
            raise FileNotFoundError(f"Dataset descriptor not found: {data}")
        metrics = self.model.val(data=str(data), split=split, imgsz=self.image_size,
                                 device=self.device, **kwargs)
        return extract_seg_metrics(metrics)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the current model weights to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path


# ─────────────────────────── segmentation helpers ───────────────────────────


def extract_seg_metrics(metrics: Any) -> Dict[str, float]:
    """Extract box + mask metrics from an Ultralytics segmentation val result."""
    return {
        "box_mAP50": float(metrics.box.map50),
        "box_mAP50-95": float(metrics.box.map),
        "mask_mAP50": float(metrics.seg.map50),
        "mask_mAP50-95": float(metrics.seg.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "fitness": float(metrics.fitness),
    }


def mask_to_polygon(mask: np.ndarray, min_points: int = 3) -> Optional[np.ndarray]:
    """Largest external contour of a binary mask as an (N, 2) float32 polygon."""
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return contour if len(contour) >= min_points else None


def _union_results(results: List[SegmentationResult]) -> Optional[SegmentationResult]:
    """Merge several part results into one whole-animal result."""
    if not results:
        return None
    union = np.zeros_like(results[0].binary_mask)
    for res in results:
        union = np.maximum(union, res.binary_mask)
    polygon = mask_to_polygon(union)
    if polygon is None:
        return None
    ys, xs = np.nonzero(union)
    return SegmentationResult(
        class_id=-1,
        class_name="animal",
        confidence=max(r.confidence for r in results),
        binary_mask=union,
        polygon=polygon,
        box_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        source="model",
    )


def estimate_body_parts(mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Approximate head / torso / legs / tail masks from a whole-animal mask.

    Deterministic geometric decomposition assuming a side-view quadruped:
      - legs:  the lower band of the animal's bounding box (bottom 38%)
      - head:  the front end band (22% of the width) above the legs; the
        front is the end whose silhouette reaches higher (head/neck)
      - tail:  the opposite end band (12% of the width) above the legs
      - torso: the remaining body pixels

    This estimation is only used when the loaded checkpoint has no body-part
    classes; a part-trained model (``segmentation.classes``) supersedes it.

    Args:
        mask: (H, W) binary mask, uint8 {0, 255} or boolean.

    Returns:
        Dict of part name -> (H, W) uint8 mask; parts with no pixels are omitted.
    """
    binary = (np.asarray(mask) > 127).astype(np.uint8)
    if binary.sum() == 0:
        return {}

    # Drop speck components (< 2% of the largest) but keep genuine body
    # regions that a slightly imperfect mask may leave disconnected.
    n_labels, labels = cv2.connectedComponents(binary)
    if n_labels > 2:
        sizes = np.bincount(labels.ravel())[1:]
        keep = np.flatnonzero(sizes >= 0.02 * sizes.max()) + 1
        binary = np.isin(labels, keep).astype(np.uint8)

    ys, xs = np.nonzero(binary)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    height = max(y1 - y0 + 1, 1)
    width = max(x1 - x0 + 1, 1)

    rows = np.arange(binary.shape[0])[:, None]
    cols = np.arange(binary.shape[1])[None, :]

    legs_top = y0 + int(0.62 * height)
    legs = binary & (rows >= legs_top)
    body = binary & (rows < legs_top)

    # Head end = the side whose silhouette reaches higher (head/neck posture).
    end_w = max(int(0.22 * width), 1)
    left_cols = binary[:, x0:x0 + end_w]
    right_cols = binary[:, max(x1 - end_w + 1, 0):x1 + 1]
    left_top = np.nonzero(left_cols.any(axis=1))[0]
    right_top = np.nonzero(right_cols.any(axis=1))[0]
    head_on_left = (left_top.min() if left_top.size else y1) <= \
                   (right_top.min() if right_top.size else y1)

    tail_w = max(int(0.12 * width), 1)
    if head_on_left:
        head = body & (cols <= x0 + end_w)
        tail = body & (cols >= x1 - tail_w)
    else:
        head = body & (cols >= x1 - end_w)
        tail = body & (cols <= x0 + tail_w)
    torso = body & ~head.astype(bool) & ~tail.astype(bool)

    parts = {"head": head, "torso": torso, "legs": legs, "tail": tail}
    return {name: (part * 255).astype(np.uint8)
            for name, part in parts.items() if np.count_nonzero(part)}


def create_overlay(
    image: np.ndarray,
    results: List[SegmentationResult],
    alpha: float = 0.45,
    draw_labels: bool = True,
) -> np.ndarray:
    """
    Blend segmentation masks over an image and outline each polygon.

    Args:
        image: (H, W, 3) image (color order is preserved in the output).
        results: Segmentation results to draw.
        alpha: Mask blend weight in [0, 1].
        draw_labels: Draw "<class> <confidence>" next to each instance.

    Returns:
        Annotated copy of the image.
    """
    overlay = image.copy()
    painted = image.copy()
    for res in results:
        color = PART_COLORS.get(res.class_name, _FALLBACK_COLOR)
        painted[res.binary_mask > 127] = color
    overlay = cv2.addWeighted(painted, alpha, overlay, 1.0 - alpha, 0)

    for res in results:
        color = PART_COLORS.get(res.class_name, _FALLBACK_COLOR)
        contour = np.round(res.polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [contour], isClosed=True, color=color, thickness=2)
        if draw_labels:
            x1, y1 = int(res.box_xyxy[0]), int(res.box_xyxy[1])
            label = f"{res.class_name} {res.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(overlay, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def export_segmentation(
    results: List[SegmentationResult],
    out_dir: Union[str, Path],
    stem: str,
    image: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Export segmentation artifacts for one image.

    Writes:
      - ``<stem>_<class>_<i>.png``   binary mask per instance
      - ``<stem>_polygons.json``     class / confidence / polygon per instance
      - ``<stem>_overlay.jpg``       overlay visualization (when ``image`` given)

    Returns:
        Manifest dict with the written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"masks": [], "polygons": None, "overlay": None}

    for i, res in enumerate(results):
        mask_path = out_dir / f"{stem}_{res.class_name}_{i}.png"
        cv2.imwrite(str(mask_path), res.binary_mask)
        manifest["masks"].append(str(mask_path))

    polygons_path = out_dir / f"{stem}_polygons.json"
    with open(polygons_path, "w") as f:
        json.dump([res.to_dict() for res in results], f)
    manifest["polygons"] = str(polygons_path)

    if image is not None and results:
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), create_overlay(image, results))
        manifest["overlay"] = str(overlay_path)
    return manifest


# ════════════════════════════════════════════════════════════════════════
#  Legacy U-Net fallback segmentor (preserved — used by inference.py)
# ════════════════════════════════════════════════════════════════════════


class DiceBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCE, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss

class UNetResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.encoder0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.encoder1 = nn.Sequential(base_model.maxpool, base_model.layer1)
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._block(128, 64)

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder0 = self._block(32, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.upconv4(e4)
        # Handle shape mismatch due to pooling
        if d4.shape != e3.shape:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:])
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        if d3.shape != e2.shape:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:])
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        if d2.shape != e1.shape:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:])
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        if d1.shape != e0.shape:
            d1 = nn.functional.interpolate(d1, size=e0.shape[2:])
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.decoder1(d1)

        d0 = self.upconv0(d1)
        d0 = self.decoder0(d0)

        out = self.final_conv(d0)
        # Match input size exactly
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:])

        return out

def calculate_metrics(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)

    dice = (2 * intersection + 1e-6) / (pred.float().sum((1, 2)) + target.float().sum((1, 2)) + 1e-6)

    pixel_acc = (pred == target).float().mean((1, 2))

    return iou.mean().item(), dice.mean().item(), pixel_acc.mean().item()

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device, save_images=False):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    dice_scores = []
    acc_scores = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="Validation")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            iou, dice, acc = calculate_metrics(outputs, masks)
            iou_scores.append(iou)
            dice_scores.append(dice)
            acc_scores.append(acc)

            if save_images and i < 5:
                # Save predicted mask overlays
                for j in range(min(4, images.size(0))):
                    img = images[j].cpu().numpy().transpose(1, 2, 0)
                    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)

                    pred_mask = (torch.sigmoid(outputs[j]).cpu().numpy()[0] > 0.5) * 255
                    pred_mask = pred_mask.astype(np.uint8)

                    # Create colored overlay
                    overlay = img.copy()
                    overlay[pred_mask == 255] = [0, 255, 0] # Green mask
                    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

                    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(OUTPUTS_DIR / f"test_pred_batch{i}_img{j}.jpg"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return running_loss / len(loader), np.mean(iou_scores), np.mean(dice_scores), np.mean(acc_scores)

def main():
    """Legacy U-Net training entry point (fallback segmentor)."""
    from utils.dataset_loader import LivestockDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = LivestockDataset(PROCESSED_DIR.parent, split="train", mode="segmentation")
    val_dataset = LivestockDataset(PROCESSED_DIR.parent, split="val", mode="segmentation")
    test_dataset = LivestockDataset(PROCESSED_DIR.parent, split="test", mode="segmentation")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = UNetResNet34().to(device)
    criterion = DiceBCE()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_iou = 0.0
    epochs = 80

    print("Starting U-Net Training...")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, val_dice, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f} | V-IoU: {val_iou:.4f} | V-Dice: {val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "unet_best.pth")
            print("Saved best model.")

    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "unet_best.pth"))
    test_loss, test_iou, test_dice, test_acc = validate(model, test_loader, criterion, device, save_images=True)

    metrics_df = pd.DataFrame([{
        'Test Loss': test_loss,
        'Mean IoU': test_iou,
        'Dice Score': test_dice,
        'Pixel Accuracy': test_acc
    }])
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"Test metrics saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()
