"""
Breed Classifier Model for the AI Cattle Analysis System.

Compares three architectures:
  1. EfficientNetV2-S
  2. ConvNeXt-Tiny  
  3. Swin Transformer Tiny

Returns top-5 breed predictions with individual confidence scores.
Replaces the naive color-histogram breed matching from the original system.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


# Default breed class names (cattle breeds from our database)
DEFAULT_BREED_CLASSES = [
    "Holstein Friesian", "Jersey", "Angus", "Hereford", "Brahman",
    "Charolais", "Simmental", "Limousin", "Gir", "Sahiwal",
    "Nellore", "Brown Swiss", "Guernsey", "Highland", "Wagyu",
    "Dexter", "Texas Longhorn", "Ayrshire", "Galloway", "Red Poll"
]


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
