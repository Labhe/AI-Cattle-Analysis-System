"""
Species Classifier Model for the AI Cattle Analysis System.

Uses EfficientNetV2-S with transfer learning to classify species.
This is a dedicated secondary classifier that confirms/corrects
the species detected by YOLO, preventing cow→dog misclassifications.

Classes: Cow, Bull, Buffalo, Yak, Ox, Goat, Sheep, Horse, Camel, Dog, Cat, Human
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import List, Tuple, Optional


SPECIES_CLASSES = [
    "Cow", "Bull", "Buffalo", "Yak", "Ox",
    "Goat", "Sheep", "Horse", "Camel",
    "Dog", "Cat", "Human"
]

# Which species are livestock (used for filtering)
LIVESTOCK_SPECIES = {"Cow", "Bull", "Buffalo", "Yak", "Ox", "Goat", "Sheep", "Horse", "Camel", "Pig"}


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
