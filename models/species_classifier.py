"""
Species Classifier Models for the AI Cattle Analysis System (Phase 5).

Dedicated secondary classifier that confirms/corrects the species detected by
YOLO, preventing cow→dog misclassifications. Runs on the detected animal ROI
after detection and segmentation.

Supported backbones (configurable via ``species_classifier.model_name``):
  - EfficientNetV2-S (default)
  - ConvNeXt-Tiny
  - Swin Transformer Tiny

All backbones support ImageNet transfer learning and share the same
Dropout → Linear classification head, so checkpoints remain interchangeable
per architecture. Class names come from the config
(``species.species_classifier_classes``) and are easily extendable — add a
class there, bump ``species_classifier.num_classes``, and retrain.

:class:`SpeciesPredictor` is the high-level inference wrapper (mirrors the
Phase 3/4 ``LivestockDetector`` / ``LivestockSegmentor`` pattern): it loads
the fine-tuned checkpoint, preprocesses raw numpy images (validating them
gracefully), and returns the predicted species with confidence and top-k
predictions.

Usage:
    from models.species_classifier import SpeciesPredictor

    predictor = SpeciesPredictor()          # config-driven
    result = predictor.predict(roi_rgb)     # {'species', 'confidence', 'top_k'}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from models.detection import load_config, select_device  # reuse Phase 3 utilities

BASE_DIR = Path(__file__).resolve().parents[1]

SPECIES_CLASSES = [
    "Cow", "Bull", "Buffalo", "Yak", "Ox",
    "Goat", "Sheep", "Horse", "Camel",
    "Dog", "Cat", "Human"
]

# Which species are livestock (used for filtering)
LIVESTOCK_SPECIES = {"Cow", "Bull", "Buffalo", "Yak", "Ox", "Goat", "Sheep", "Horse", "Camel", "Pig"}

SUPPORTED_ARCHITECTURES = ("efficientnet_v2_s", "convnext_tiny", "swin_t")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_species_classes(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Class names ordered by id from ``species.species_classifier_classes``,
    falling back to the module-level :data:`SPECIES_CLASSES`.
    """
    config = config if config is not None else load_config()
    declared = config.get("species", {}).get("species_classifier_classes", {})
    if not declared:
        return list(SPECIES_CLASSES)
    return [str(declared[k]) for k in sorted(declared, key=int)]


class SpeciesClassifier(nn.Module):
    """
    EfficientNetV2-S based species classifier.

    Architecture:
        - Backbone: EfficientNetV2-S (pretrained on ImageNet)
        - Head: Dropout(0.3) → Linear(1280, num_classes)

    Input: (B, 3, 224, 224) normalized image tensor
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 12, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = "efficientnet_v2_s"

        # Load pretrained EfficientNetV2-S
        if pretrained:
            self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_v2_s(weights=None)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.backbone(x)

    def predict_top_k(
        self,
        x: torch.Tensor,
        k: int = 3,
        class_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k species with confidence scores.

        Args:
            x: Input tensor (1, 3, 224, 224)
            k: Number of top predictions
            class_names: List of class names (defaults to SPECIES_CLASSES)

        Returns:
            List of (species_name, confidence) tuples.
        """
        if class_names is None:
            class_names = SPECIES_CLASSES

        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_k_probs, top_k_indices = torch.topk(probs, min(k, len(class_names)))

        results = []
        for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
            name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            results.append((name, float(prob)))

        return results

    def is_livestock(self, species: str) -> bool:
        """Check if the predicted species is livestock."""
        return species in LIVESTOCK_SPECIES


class ConvNeXtSpeciesClassifier(nn.Module):
    """ConvNeXt-Tiny based species classifier (same head shape as the default)."""

    def __init__(self, num_classes: int = 12, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

        self.num_classes = num_classes
        self.architecture = "convnext_tiny"
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.backbone(x)


class SwinSpeciesClassifier(nn.Module):
    """Swin Transformer Tiny based species classifier."""

    def __init__(self, num_classes: int = 12, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights

        self.num_classes = num_classes
        self.architecture = "swin_t"
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.backbone(x)


def build_species_classifier(
    architecture: str = "efficientnet_v2_s",
    num_classes: int = 12,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a species classifier from one of the supported architectures.

    Args:
        architecture: One of :data:`SUPPORTED_ARCHITECTURES`.
        num_classes: Number of species classes.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout rate for the classifier head.

    Raises:
        ValueError: If ``architecture`` is not supported.
    """
    if architecture == "efficientnet_v2_s":
        return SpeciesClassifier(num_classes, dropout, pretrained)
    if architecture == "convnext_tiny":
        return ConvNeXtSpeciesClassifier(num_classes, dropout, pretrained)
    if architecture == "swin_t":
        return SwinSpeciesClassifier(num_classes, dropout, pretrained)
    raise ValueError(
        f"Unknown architecture: {architecture!r}. Choose from {SUPPORTED_ARCHITECTURES}"
    )


class SpeciesPredictor:
    """
    High-level, config-driven species inference wrapper.

    Loads the fine-tuned checkpoint (``species_classifier.weights``), builds
    the configured backbone, auto-selects CPU/CUDA, and exposes a single
    :meth:`predict` that accepts raw numpy images.

    Args:
        config: Parsed ``model_config.yaml`` dict; loaded from disk if omitted.
        weights: Explicit checkpoint path (state dict). Defaults to the config
            path; a missing checkpoint raises ``FileNotFoundError``.
        architecture: Override for ``species_classifier.model_name``.
        device: 'auto' (default), 'cpu', 'cuda', or an explicit CUDA index.

    Raises:
        FileNotFoundError: If no trained checkpoint is available.
        ValueError: If the configured architecture is unsupported.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weights: Optional[Union[str, Path]] = None,
        architecture: Optional[str] = None,
        device: str = "auto",
    ):
        self.config = config if config is not None else load_config()
        clf_cfg = self.config.get("species_classifier", {})
        self.device = select_device(device or self.config.get("inference", {}).get("device", "auto"))
        self.class_names = load_species_classes(self.config)
        self.image_size = int(clf_cfg.get("image_size", 224))
        self.top_k = int(clf_cfg.get("top_k", 5))
        self.min_confidence = float(clf_cfg.get("min_confidence", 0.5))
        self.architecture = architecture or clf_cfg.get("model_name", "efficientnet_v2_s")

        weights = Path(weights) if weights is not None \
            else BASE_DIR / clf_cfg.get("weights", "checkpoints/species_classifier_best.pth")
        if not weights.exists():
            raise FileNotFoundError(
                f"Species classifier checkpoint not found: {weights} "
                f"(train one with training/train_species_classifier.py)"
            )
        self.weights_path = weights

        self.model = build_species_classifier(
            architecture=self.architecture,
            num_classes=len(self.class_names),
            dropout=float(clf_cfg.get("training", {}).get("dropout", 0.3)),
            pretrained=False,
        )
        state_dict = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self._mean, self._std = mean, std

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Validate and convert a raw image array to a normalized model input.

        Accepts (H, W, 3) RGB, (H, W) grayscale, or (H, W, 4) RGBA uint8/float
        arrays.

        Raises:
            ValueError: If the image is None, empty, or has an unsupported shape.
        """
        import cv2

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image: empty or not a numpy array")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape} (expected HxWx3)")
        if min(image.shape[:2]) < 8:
            raise ValueError(f"Image too small to classify: {image.shape[:2]}")

        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        if tensor.max() > 1.5:  # uint8-range input
            tensor = tensor / 255.0
        tensor = (tensor - self._mean) / self._std
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Classify the species in an RGB image array.

        Returns:
            Dict with:
              - ``species``: top-1 class name
              - ``confidence``: top-1 softmax probability
              - ``top_k``: list of (species, confidence) tuples (top-5 default)

        Raises:
            ValueError: If the image is invalid (see :meth:`preprocess`).
        """
        k = min(top_k or self.top_k, len(self.class_names))
        tensor = self.preprocess(image)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idx = torch.topk(probs, k)
        predictions = [
            (self.class_names[int(i)], float(p))
            for p, i in zip(top_probs.cpu(), top_idx.cpu())
        ]
        return {
            "species": predictions[0][0],
            "confidence": predictions[0][1],
            "top_k": predictions,
        }

    def is_livestock(self, species: str) -> bool:
        """Check if a species name is livestock."""
        return species in LIVESTOCK_SPECIES
