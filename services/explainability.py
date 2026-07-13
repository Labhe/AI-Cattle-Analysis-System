"""
Explainability Service for the AI Cattle Analysis System.

Generates GradCAM heatmaps, attention maps, and confidence visualizations
to explain model predictions. Provides visual evidence for why the model
predicted a specific breed or species.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from pathlib import Path


class GradCAMGenerator:
    """
    Generate Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
    for CNN-based classifiers (EfficientNet, ConvNeXt, etc.).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The classification model.
            target_layer: The convolutional layer to compute GradCAM on
                         (usually the last conv layer before the classifier head).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a GradCAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed input image tensor (1, C, H, W).
            target_class: Class index to generate heatmap for.
                         If None, uses the predicted class.

        Returns:
            Heatmap as a numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback: return blank heatmap
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original image (H, W, 3) in RGB.
        heatmap: Heatmap array (H, W) with values in [0, 1].
        alpha: Blending factor for the heatmap overlay.
        colormap: OpenCV colormap to apply.

    Returns:
        Blended image with heatmap overlay (H, W, 3) in RGB.
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    output = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return output


def create_confidence_bar_chart(
    labels: List[str],
    confidences: List[float],
    width: int = 400,
    height: int = 250,
    bar_color: Tuple[int, int, int] = (59, 130, 246),
    bg_color: Tuple[int, int, int] = (15, 23, 42),
) -> np.ndarray:
    """
    Create a confidence bar chart as an image (no matplotlib dependency).

    Args:
        labels: List of class/breed names.
        confidences: List of confidence values (0-1).
        width: Image width.
        height: Image height.

    Returns:
        Bar chart as numpy array (H, W, 3) in RGB.
    """
    chart = np.full((height, width, 3), bg_color, dtype=np.uint8)

    if not labels or not confidences:
        return chart

    n = min(len(labels), 5)  # Max 5 bars
    bar_height = max(15, (height - 40) // n - 10)
    max_bar_width = width - 160

    for i in range(n):
        y = 20 + i * (bar_height + 10)
        conf = min(1.0, max(0.0, confidences[i]))
        bar_w = int(conf * max_bar_width)

        # Label
        label_text = labels[i][:20]
        cv2.putText(
            chart, label_text, (10, y + bar_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

        # Bar
        bar_x = 130
        cv2.rectangle(
            chart, (bar_x, y), (bar_x + bar_w, y + bar_height),
            bar_color, -1
        )

        # Percentage text
        pct_text = f"{conf * 100:.1f}%"
        cv2.putText(
            chart, pct_text, (bar_x + bar_w + 5, y + bar_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1
        )

    return chart


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.35,
    draw_contour: bool = True,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Create a segmentation mask overlay on the original image.

    Args:
        image: Original image (H, W, 3) in RGB.
        mask: Binary mask (H, W) with 0/255 or boolean values.
        color: Fill color for the mask overlay.
        alpha: Blending factor.
        draw_contour: Whether to draw contour lines.

    Returns:
        Image with segmentation overlay (H, W, 3) in RGB.
    """
    # Ensure mask is the right size
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_bool = mask > 127 if mask.max() > 1 else mask > 0.5

    overlay = image.copy()
    overlay[mask_bool] = color
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    if draw_contour:
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, contour_color, contour_thickness)

    return result
