"""
Breed Classifier Models for the AI Cattle Analysis System (Phase 6).

Fine-grained cattle breed classification that runs as the final
classification stage of the pipeline:

    Detection → Segmentation → Species Classification → Breed Classification

Supported backbones (configurable via ``breed_classifier.model_name``):
  1. EfficientNetV2-S
  2. ConvNeXt-Tiny (default)
  3. Swin Transformer Tiny

All backbones support ImageNet transfer learning. :class:`BreedPredictor` is
the high-level inference wrapper (mirrors ``SpeciesPredictor`` from Phase 5):
it loads the fine-tuned checkpoint, resolves the trained breed class list
(``checkpoints/breed_classes.json`` written by the trainer, falling back to
the defaults below), validates raw numpy images gracefully, and **gates on
species** — breed prediction is only attempted for supported livestock
species (``breed_classifier.supported_species``); other species get an
explanatory message instead of a forced prediction.

Returns top-5 breed predictions with individual confidence scores.
Replaces the naive color-histogram breed matching from the original system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.detection import load_config, select_device  # reuse Phase 3 utilities

BASE_DIR = Path(__file__).resolve().parents[1]

SUPPORTED_ARCHITECTURES = ("efficientnet_v2_s", "convnext_tiny", "swin_t")

# Species for which the (cattle-)breed classifier is applicable. Extend via
# breed_classifier.supported_species in the config after training on more.
DEFAULT_SUPPORTED_SPECIES = ("Cow", "Bull", "Ox", "Cattle")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default breed class names (cattle breeds from our database)
DEFAULT_BREED_CLASSES = [
    "Holstein Friesian", "Jersey", "Angus", "Hereford", "Brahman",
    "Charolais", "Simmental", "Limousin", "Gir", "Sahiwal",
    "Nellore", "Brown Swiss", "Guernsey", "Highland", "Wagyu",
    "Dexter", "Texas Longhorn", "Ayrshire", "Galloway", "Red Poll"
]


def load_breed_metadata(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read the trainer-written metadata file (``breed_classifier.classes_file``).

    Supports both the metadata dict format ``{"classes": [...],
    "architecture": "..."}`` and the legacy bare-list format.

    Returns:
        Dict with ``classes`` (list or None) and ``architecture`` (str or None).
    """
    config = config if config is not None else load_config()
    clf_cfg = config.get("breed_classifier", {})
    classes_file = BASE_DIR / clf_cfg.get("classes_file", "checkpoints/breed_classes.json")
    if classes_file.exists():
        try:
            with open(classes_file, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and payload.get("classes"):
                return {"classes": [str(c) for c in payload["classes"]],
                        "architecture": payload.get("architecture")}
            if isinstance(payload, list) and payload:
                return {"classes": [str(c) for c in payload], "architecture": None}
        except (json.JSONDecodeError, OSError):
            pass
    return {"classes": None, "architecture": None}


def load_breed_classes(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Resolve the breed class list, in priority order:
      1. ``breed_classifier.classes`` declared directly in the config
      2. The metadata file written by the trainer (``breed_classifier.classes_file``)
      3. :data:`DEFAULT_BREED_CLASSES`
    """
    config = config if config is not None else load_config()
    declared = config.get("breed_classifier", {}).get("classes")
    if declared:
        return [str(c) for c in declared]
    from_file = load_breed_metadata(config)["classes"]
    return from_file if from_file else list(DEFAULT_BREED_CLASSES)


def build_breed_classifier(
    architecture: str = "convnext_tiny",
    num_classes: int = 20,
    pretrained: bool = True,
    dropout: float = 0.4,
) -> nn.Module:
    """
    Build a breed classifier from one of the supported architectures.
    
    Args:
        architecture: One of 'efficientnet_v2_s', 'convnext_tiny', 'swin_t'
        num_classes: Number of breed classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate for the classifier head
    
    Returns:
        PyTorch model ready for training or inference.
    """
    if architecture == "efficientnet_v2_s":
        return EfficientNetV2BreedClassifier(num_classes, dropout, pretrained)
    elif architecture == "convnext_tiny":
        return ConvNeXtBreedClassifier(num_classes, dropout, pretrained)
    elif architecture == "swin_t":
        return SwinBreedClassifier(num_classes, dropout, pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'efficientnet_v2_s', 'convnext_tiny', or 'swin_t'")


class EfficientNetV2BreedClassifier(nn.Module):
    """EfficientNetV2-S based breed classifier."""

    def __init__(self, num_classes: int = 20, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )
        self.architecture = "efficientnet_v2_s"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ConvNeXtBreedClassifier(nn.Module):
    """ConvNeXt-Tiny based breed classifier."""

    def __init__(self, num_classes: int = 20, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )
        self.architecture = "convnext_tiny"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SwinBreedClassifier(nn.Module):
    """Swin Transformer Tiny based breed classifier."""

    def __init__(self, num_classes: int = 20, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights
        
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )
        self.architecture = "swin_t"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def predict_breed_top_k(
    model: nn.Module,
    input_tensor: torch.Tensor,
    k: int = 5,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, any]]:
    """
    Predict top-k breeds with confidence scores.
    
    Args:
        model: Breed classifier model.
        input_tensor: Preprocessed input (1, 3, 224, 224).
        k: Number of top predictions.
        class_names: List of breed names.
    
    Returns:
        List of dicts: [{"breed": str, "confidence": float}, ...]
    """
    if class_names is None:
        class_names = DEFAULT_BREED_CLASSES

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_k_probs, top_k_indices = torch.topk(probs, min(k, len(class_names)))

    results = []
    for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
        name = class_names[idx] if idx < len(class_names) else f"Breed_{idx}"
        results.append({
            "breed": name,
            "confidence": round(float(prob) * 100, 1),
        })

    return results


class BreedPredictor:
    """
    High-level, config-driven breed inference wrapper.

    Loads the fine-tuned checkpoint (``breed_classifier.weights``), builds the
    configured backbone, auto-selects CPU/CUDA, and exposes :meth:`predict`
    for raw numpy images plus :meth:`predict_for_species`, which enforces the
    pipeline contract that breed classification only runs for supported
    livestock species (after detection, segmentation, and species
    classification).

    Args:
        config: Parsed ``model_config.yaml`` dict; loaded from disk if omitted.
        weights: Explicit checkpoint path (state dict). Defaults to the config
            path; a missing checkpoint raises ``FileNotFoundError``.
        architecture: Override for ``breed_classifier.model_name``.
        class_names: Override for the trained breed class list.
        device: 'auto' (default), 'cpu', 'cuda', or an explicit CUDA index.

    Raises:
        FileNotFoundError: If no trained checkpoint is available.
        ValueError: If the configured architecture is unsupported.
        RuntimeError: If the checkpoint does not match the configured
            architecture/class count.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weights: Optional[Union[str, Path]] = None,
        architecture: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        device: str = "auto",
    ):
        self.config = config if config is not None else load_config()
        clf_cfg = self.config.get("breed_classifier", {})
        self.device = select_device(device or self.config.get("inference", {}).get("device", "auto"))
        self.class_names = class_names or load_breed_classes(self.config)
        self.image_size = int(clf_cfg.get("image_size", 224))
        self.top_k = int(clf_cfg.get("top_k", 5))
        self.min_confidence = float(clf_cfg.get("min_confidence", 0.3))
        # Architecture priority: explicit arg → trainer metadata (what the
        # checkpoint was actually trained with) → config default.
        metadata = load_breed_metadata(self.config)
        self.architecture = (architecture or metadata["architecture"]
                             or clf_cfg.get("model_name", "convnext_tiny"))
        self.supported_species = {
            str(s).lower()
            for s in clf_cfg.get("supported_species", DEFAULT_SUPPORTED_SPECIES)
        }

        weights = Path(weights) if weights is not None \
            else BASE_DIR / clf_cfg.get("weights", "checkpoints/breed_classifier_best.pth")
        if not weights.exists():
            raise FileNotFoundError(
                f"Breed classifier checkpoint not found: {weights} "
                f"(train one with training/train_breed_classifier.py)"
            )
        self.weights_path = weights

        self.model = build_breed_classifier(
            architecture=self.architecture,
            num_classes=len(self.class_names),
            dropout=float(clf_cfg.get("training", {}).get("dropout", 0.4)),
            pretrained=False,
        )
        try:
            state_dict = torch.load(weights, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint {weights} does not match architecture "
                f"{self.architecture!r} with {len(self.class_names)} classes: {e}"
            ) from e
        self.model.to(self.device)
        self.model.eval()

        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def supports_species(self, species: str) -> bool:
        """True if breed classification is applicable to this species."""
        return str(species).strip().lower() in self.supported_species

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Validate and convert a raw image array to a normalized model input.

        Accepts (H, W, 3) RGB, (H, W) grayscale, or (H, W, 4) RGBA arrays.

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
        Classify the breed in an RGB image array (no species gating).

        Returns:
            Dict with:
              - ``breed``: top-1 breed name
              - ``confidence``: top-1 softmax probability (0–1)
              - ``top_k``: list of (breed, confidence) tuples (top-5 default)

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
            "breed": predictions[0][0],
            "confidence": predictions[0][1],
            "top_k": predictions,
        }

    def predict_for_species(self, image: np.ndarray, species: str,
                            top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Species-gated breed prediction (the pipeline entry point).

        For unsupported species no prediction is forced; the result carries
        ``supported=False`` and an explanatory ``message``.

        Returns:
            Dict with ``supported`` (bool) and either the :meth:`predict`
            fields (supported) or ``message`` (unsupported).

        Raises:
            ValueError: If the image is invalid.
        """
        if not self.supports_species(species):
            return {
                "supported": False,
                "message": (
                    f"Breed classification is not available for species "
                    f"'{species}'. Supported species: "
                    + ", ".join(sorted(s.title() for s in self.supported_species))
                ),
            }
        result = self.predict(image, top_k=top_k)
        result["supported"] = True
        return result
