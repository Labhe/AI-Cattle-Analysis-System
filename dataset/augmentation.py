"""
Augmentation Pipelines for the AI Cattle Analysis System (Phase 2).

Centralized Albumentations pipelines for every training task, built for the
unified dataset produced by ``dataset/unify_datasets.py``:

  - detection       : YOLO-format bounding boxes (geometric + photometric)
  - segmentation    : binary masks (geometric + photometric)
  - classification  : species / breed classifiers (RandomResizedCrop-based)
  - regression      : weight / BCS estimation — photometric only, since scale
                      and aspect distortions would corrupt morphometric targets

Each builder accepts an ``intensity`` preset ('light' | 'medium' | 'heavy')
so the training scripts can tune augmentation strength without editing
pipeline code. Validation/test pipelines are deterministic (resize +
normalize only).

Also provides batch-level MixUp and CutMix for the classifier training loops.

Usage:
    from dataset.augmentation import build_transforms, transforms_from_config

    train_tf = build_transforms("detection", split="train", image_size=640)
    val_tf   = build_transforms("classification", split="val", image_size=224)
    tf       = transforms_from_config(config, task="breed_classifier", split="train")
"""

from typing import Any, Dict, Optional, Tuple

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VALID_TASKS = ("detection", "segmentation", "classification", "regression")

# Probability / magnitude presets shared by all task pipelines.
INTENSITY_PRESETS: Dict[str, Dict[str, float]] = {
    "light": {
        "p_flip": 0.5, "p_affine": 0.2, "p_color": 0.3, "p_blur_noise": 0.1,
        "p_weather": 0.0, "p_dropout": 0.1,
        "rotate_deg": 5.0, "translate": 0.05, "scale": 0.1, "shear_deg": 0.0,
        "brightness": 0.1, "contrast": 0.1, "hue": 5, "saturation": 15, "value": 10,
    },
    "medium": {
        "p_flip": 0.5, "p_affine": 0.4, "p_color": 0.5, "p_blur_noise": 0.2,
        "p_weather": 0.1, "p_dropout": 0.2,
        "rotate_deg": 10.0, "translate": 0.1, "scale": 0.25, "shear_deg": 5.0,
        "brightness": 0.2, "contrast": 0.2, "hue": 10, "saturation": 30, "value": 20,
    },
    "heavy": {
        "p_flip": 0.5, "p_affine": 0.6, "p_color": 0.7, "p_blur_noise": 0.35,
        "p_weather": 0.2, "p_dropout": 0.35,
        "rotate_deg": 15.0, "translate": 0.15, "scale": 0.4, "shear_deg": 10.0,
        "brightness": 0.3, "contrast": 0.3, "hue": 15, "saturation": 40, "value": 30,
    },
}


def _preset(intensity: str) -> Dict[str, float]:
    if intensity not in INTENSITY_PRESETS:
        raise ValueError(f"Unknown intensity '{intensity}'. "
                         f"Choose from {sorted(INTENSITY_PRESETS)}")
    return INTENSITY_PRESETS[intensity]


def _photometric_block(p: Dict[str, float]) -> list:
    """Color / blur / noise / weather transforms shared by all tasks."""
    return [
        A.RandomBrightnessContrast(brightness_limit=p["brightness"],
                                   contrast_limit=p["contrast"], p=p["p_color"]),
        A.HueSaturationValue(hue_shift_limit=int(p["hue"]),
                             sat_shift_limit=int(p["saturation"]),
                             val_shift_limit=int(p["value"]), p=p["p_color"] * 0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.GaussNoise(std_range=(0.02, 0.1)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=p["p_blur_noise"]),
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1)),
            A.RandomFog(fog_coef_range=(0.1, 0.3)),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.3), src_radius=100),
        ], p=p["p_weather"]),
        A.CLAHE(clip_limit=2.0, p=p["p_color"] * 0.3),
    ]


def _dropout_block(p: Dict[str, float]) -> list:
    """Occlusion simulation (fences, feeders, other animals)."""
    return [
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(0.02, 0.08),
                        hole_width_range=(0.02, 0.08),
                        fill=0, p=p["p_dropout"]),
    ]


def _resize_pad_block(image_size: int) -> list:
    """Aspect-preserving letterbox resize used by detection/segmentation."""
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ]


def _finalize(transforms: list, normalize: bool, to_tensor: bool) -> list:
    if normalize:
        transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    if to_tensor:
        transforms.append(ToTensorV2())
    return transforms


# ───────────────────────────── task pipelines ─────────────────────────────


def get_detection_transforms(split: str = "train", image_size: int = 640,
                             intensity: str = "medium", normalize: bool = True,
                             to_tensor: bool = True) -> A.Compose:
    """Detection pipeline with YOLO-format bbox handling."""
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"],
                               min_visibility=0.2, min_area=16)
    if split != "train":
        transforms = _resize_pad_block(image_size)
        return A.Compose(_finalize(transforms, normalize, to_tensor), bbox_params=bbox_params)

    p = _preset(intensity)
    transforms = _resize_pad_block(image_size) + [
        A.HorizontalFlip(p=p["p_flip"]),
        A.Affine(scale=(1 - p["scale"], 1 + p["scale"]),
                 translate_percent=(-p["translate"], p["translate"]),
                 rotate=(-p["rotate_deg"], p["rotate_deg"]),
                 shear=(-p["shear_deg"], p["shear_deg"]),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=p["p_affine"]),
        *_photometric_block(p),
        *_dropout_block(p),
    ]
    return A.Compose(_finalize(transforms, normalize, to_tensor), bbox_params=bbox_params)


def get_segmentation_transforms(split: str = "train", image_size: int = 640,
                                intensity: str = "medium", normalize: bool = True,
                                to_tensor: bool = True) -> A.Compose:
    """Segmentation pipeline; masks follow every geometric transform."""
    if split != "train":
        return A.Compose(_finalize(_resize_pad_block(image_size), normalize, to_tensor))

    p = _preset(intensity)
    transforms = _resize_pad_block(image_size) + [
        A.HorizontalFlip(p=p["p_flip"]),
        A.Affine(scale=(1 - p["scale"], 1 + p["scale"]),
                 translate_percent=(-p["translate"], p["translate"]),
                 rotate=(-p["rotate_deg"], p["rotate_deg"]),
                 shear=(-p["shear_deg"], p["shear_deg"]),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=p["p_affine"]),
        A.ElasticTransform(alpha=50, sigma=7, p=p["p_affine"] * 0.3),
        *_photometric_block(p),
    ]
    return A.Compose(_finalize(transforms, normalize, to_tensor))


def get_classification_transforms(split: str = "train", image_size: int = 224,
                                  intensity: str = "medium", normalize: bool = True,
                                  to_tensor: bool = True) -> A.Compose:
    """Species / breed classifier pipeline (RandomResizedCrop for train)."""
    if split != "train":
        transforms = [
            A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
            A.CenterCrop(height=image_size, width=image_size),
        ]
        return A.Compose(_finalize(transforms, normalize, to_tensor))

    p = _preset(intensity)
    transforms = [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0),
                            ratio=(0.75, 1.333)),
        A.HorizontalFlip(p=p["p_flip"]),
        A.Affine(rotate=(-p["rotate_deg"], p["rotate_deg"]),
                 shear=(-p["shear_deg"], p["shear_deg"]),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=p["p_affine"] * 0.5),
        *_photometric_block(p),
        *_dropout_block(p),
    ]
    return A.Compose(_finalize(transforms, normalize, to_tensor))


def get_regression_transforms(split: str = "train", image_size: int = 224,
                              intensity: str = "medium", normalize: bool = True,
                              to_tensor: bool = True) -> A.Compose:
    """
    Weight / BCS regression pipeline.

    Deliberately photometric-only (plus horizontal flip): scale, crop, and
    aspect distortions change the animal's apparent size and proportions,
    which would decouple the image from its weight/BCS target.
    """
    transforms = [A.Resize(height=image_size, width=image_size)]
    if split == "train":
        p = _preset(intensity)
        transforms += [A.HorizontalFlip(p=p["p_flip"]), *_photometric_block(p)]
    return A.Compose(_finalize(transforms, normalize, to_tensor))


_BUILDERS = {
    "detection": get_detection_transforms,
    "segmentation": get_segmentation_transforms,
    "classification": get_classification_transforms,
    "regression": get_regression_transforms,
}

# Config section -> (task, default image size) for transforms_from_config().
_CONFIG_SECTIONS = {
    "detection": ("detection", 640),
    "segmentation": ("segmentation", 640),
    "species_classifier": ("classification", 224),
    "breed_classifier": ("classification", 224),
    "bcs_regressor": ("regression", 224),
    "weight_regressor": ("regression", 224),
}


def build_transforms(task: str, split: str = "train", **kwargs: Any) -> A.Compose:
    """Build a pipeline by task name; see the task-specific builders."""
    if task not in _BUILDERS:
        raise ValueError(f"Unknown task '{task}'. Choose from {VALID_TASKS}")
    return _BUILDERS[task](split=split, **kwargs)


def transforms_from_config(config: Dict[str, Any], task: str,
                           split: str = "train", **kwargs: Any) -> A.Compose:
    """
    Build a pipeline for a model section of ``configs/model_config.yaml``
    (e.g. 'breed_classifier'), pulling ``image_size`` from that section.
    """
    if task not in _CONFIG_SECTIONS:
        raise ValueError(f"Unknown config section '{task}'. "
                         f"Choose from {sorted(_CONFIG_SECTIONS)}")
    base_task, default_size = _CONFIG_SECTIONS[task]
    image_size = config.get(task, {}).get("image_size", default_size)
    kwargs.setdefault("image_size", image_size)
    return build_transforms(base_task, split=split, **kwargs)


# ───────────────────────── batch-level augmentation ─────────────────────────


def mixup_batch(images: torch.Tensor, targets: torch.Tensor,
                alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp (Zhang et al., 2018) for classifier training.

    Args:
        images: (B, C, H, W) batch.
        targets: (B,) integer class labels.
        alpha: Beta-distribution concentration; 0 disables mixing.

    Returns:
        (mixed_images, targets_a, targets_b, lam) — compute the loss as
        ``lam * criterion(out, targets_a) + (1 - lam) * criterion(out, targets_b)``.
    """
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(images: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix (Yun et al., 2019) for classifier training.

    Same return contract as :func:`mixup_batch`; ``lam`` is corrected to the
    exact pasted-area ratio.
    """
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(images.size(0), device=images.device)

    _, _, h, w = images.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy = int(torch.randint(0, h, (1,)))
    cx = int(torch.randint(0, w, (1,)))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
    return mixed, targets, targets[index], lam
