"""
Explainability Service for the AI Cattle Analysis System.

Generates GradCAM heatmaps, attention maps, and confidence visualizations
to explain model predictions. Provides visual evidence for why the model
predicted a specific breed or species.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


def _to_nchw(t: torch.Tensor, channels_last: bool) -> torch.Tensor:
    """Return a (B, C, H, W) view of a 4D activation/gradient tensor."""
    if channels_last and t.ndim == 4:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Resolve the Grad-CAM target layer for a project classifier backbone.

    Prefers the last stage of ``model.backbone.features`` (EfficientNetV2,
    ConvNeXt, Swin all expose it); otherwise falls back to the last
    ``nn.Conv2d`` / ``nn.LayerNorm`` in the module tree.

    Raises:
        ValueError: If no suitable layer can be found.
    """
    backbone = getattr(model, "backbone", model)
    features = getattr(backbone, "features", None)
    if isinstance(features, nn.Sequential) and len(features) > 0:
        return features[-1]
    last = None
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.LayerNorm)):
            last = module
    if last is None:
        raise ValueError("Could not resolve a Grad-CAM target layer for this model")
    return last


def is_channels_last(model: nn.Module) -> bool:
    """True for backbones whose feature maps are channels-last (e.g. Swin)."""
    return str(getattr(model, "architecture", "")).startswith("swin")


class GradCAMGenerator:
    """
    Generate Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
    for CNN-based classifiers (EfficientNet, ConvNeXt, etc.).

    Supports channels-last feature maps (transformers such as Swin) via the
    ``channels_last`` flag. Call :meth:`remove` to detach the hooks.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None,
                 channels_last: bool = False):
        """
        Args:
            model: The classification model.
            target_layer: The layer to compute Grad-CAM on; auto-resolved via
                :func:`get_target_layer` when omitted.
            channels_last: Set for backbones with (B, H, W, C) feature maps.
        """
        self.model = model
        self.target_layer = target_layer or get_target_layer(model)
        self.channels_last = channels_last
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        """Remove the registered hooks (call when done to avoid leaks)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

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

        activations = _to_nchw(self.activations, self.channels_last)
        gradients = _to_nchw(self.gradients, self.channels_last)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

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


# ════════════════════════════════════════════════════════════════════════
#  Attention maps, confidence heatmaps & explanations (Phase 11)
# ════════════════════════════════════════════════════════════════════════


class AttentionMapGenerator:
    """
    Gradient-free attention/saliency maps from a backbone's feature maps.

    Averages the (absolute) channel activations of the target layer to show
    which spatial regions the model attends to — works for CNNs and
    transformers (channels-last handled) without a backward pass. Complements
    Grad-CAM, which is class-specific.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None,
                 channels_last: Optional[bool] = None):
        self.model = model
        self.target_layer = target_layer or get_target_layer(model)
        self.channels_last = is_channels_last(model) if channels_last is None else channels_last
        self.activations: Optional[torch.Tensor] = None
        self._handle = self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))

    def remove(self) -> None:
        """Remove the forward hook."""
        self._handle.remove()

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Produce a normalized (H, W) attention map in [0, 1] for one image.
        """
        self.model.eval()
        self.model(input_tensor)
        if self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
        act = _to_nchw(self.activations, self.channels_last)
        attn = act.abs().mean(dim=1).squeeze().cpu().numpy()
        attn = attn - attn.min()
        if attn.max() > 0:
            attn = attn / attn.max()
        return cv2.resize(attn.astype(np.float32),
                          (input_tensor.shape[3], input_tensor.shape[2]))


def create_confidence_heatmap(
    image: np.ndarray,
    saliency: np.ndarray,
    confidence: float,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Render a confidence-weighted heatmap overlay.

    The saliency map (Grad-CAM or attention) is scaled by the prediction
    confidence so the overlay intensity reflects how certain the model is:
    a high-confidence prediction yields a vivid heatmap, a low-confidence one
    a faint map.

    Args:
        image: Original image (H, W, 3) in RGB.
        saliency: Saliency/CAM map (H, W) in [0, 1].
        confidence: Prediction confidence in [0, 1] (values >1 treated as %).
        alpha: Base blend factor.

    Returns:
        RGB image (H, W, 3) with the confidence heatmap overlaid.
    """
    conf = confidence / 100.0 if confidence > 1.0 else confidence
    conf = float(min(1.0, max(0.0, conf)))
    weighted = np.clip(saliency, 0.0, 1.0) * conf
    return overlay_heatmap(image, weighted, alpha=alpha)


def generate_prediction_explanation(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a human-readable explanation of an inference result.

    Summarizes the decision at each pipeline stage and lists the salient
    factors (confidence levels, morphometric drivers, health signals) so a
    user understands *why* the system reported these predictions.

    Args:
        analysis: Result dict from the inference pipeline.

    Returns:
        Dict with ``summary`` (str), ``factors`` (list of str), and
        ``per_stage`` (dict of stage -> explanation).
    """
    analysis = analysis or {}
    species = analysis.get("species", "Unknown")
    species_conf = analysis.get("species_confidence", 0.0)
    breed = analysis.get("breed", "Unknown")
    breed_conf = analysis.get("breed_confidence", 0.0)
    weight = analysis.get("weight_kg")
    weight_method = analysis.get("weight_method", "unknown")
    bcs = analysis.get("bcs")
    measurements = analysis.get("measurements", {}) or {}

    def band(conf: float) -> str:
        conf = conf if conf <= 1.0 else conf / 100.0
        if conf >= 0.85:
            return "high"
        if conf >= 0.6:
            return "moderate"
        return "low"

    per_stage = {
        "detection": (
            f"An animal was localized with "
            f"{analysis.get('detection', {}).get('confidence', 0)}% detection confidence."
        ),
        "species": (
            f"Classified as {species} with {band(species_conf)} confidence "
            f"({species_conf}%)."
        ),
        "breed": (
            f"Breed identified as {breed} ({breed_conf}% confidence)."
            if analysis.get("breed_method") == "classifier"
            else analysis.get("breed_message")
            or f"Breed reported as {breed} via {analysis.get('breed_method', 'fallback')}."
        ),
        "weight": (
            f"Weight estimated at {weight} kg from body measurements "
            f"({weight_method})."
        ),
        "bcs": (
            f"Body Condition Score assessed as {bcs} on the 1–5 scale."
            if bcs is not None else "BCS unavailable."
        ),
    }

    factors: List[str] = []
    if measurements:
        if measurements.get("heart_girth_px"):
            factors.append(
                f"Heart girth ({measurements['heart_girth_px']} px) and body length "
                f"({measurements.get('body_length_px', 'n/a')} px) primarily drive the weight estimate.")
        if measurements.get("body_area_px"):
            factors.append(f"Visible body area: {measurements['body_area_px']} px.")
    factors.append(f"Species decision confidence is {band(species_conf)} ({species_conf}%).")
    if analysis.get("breed_method") == "classifier":
        factors.append(f"Breed decision confidence is {band(breed_conf)} ({breed_conf}%).")

    summary = (
        f"The system identified a {species} of breed {breed}, estimated at "
        f"{weight} kg with a body condition score of {bcs}. "
        f"Species classification confidence was {band(species_conf)}."
    )
    return {"summary": summary, "factors": factors, "per_stage": per_stage}


class ExplainabilityService:
    """
    High-level orchestrator producing every explanation artifact for one
    prediction: Grad-CAM, attention map, confidence heatmap, segmentation
    overlay, and a textual prediction explanation.

    Args:
        model: The classifier to explain (species or breed model). Optional —
            when omitted, only the segmentation overlay and textual
            explanation are produced.
        device: Torch device string for tensor placement.
    """

    def __init__(self, model: Optional[nn.Module] = None, device: str = "cpu"):
        self.model = model
        self.device = device
        self.channels_last = is_channels_last(model) if model is not None else False

    def gradcam(self, input_tensor: torch.Tensor, image: np.ndarray,
                target_class: Optional[int] = None, alpha: float = 0.45) -> Dict[str, np.ndarray]:
        """Grad-CAM map + overlay for the given input (requires a model)."""
        if self.model is None:
            raise ValueError("A model is required for Grad-CAM")
        cam_gen = GradCAMGenerator(self.model, channels_last=self.channels_last)
        try:
            cam = cam_gen.generate(input_tensor.to(self.device), target_class)
        finally:
            cam_gen.remove()
        return {"map": cam, "overlay": overlay_heatmap(image, cam, alpha=alpha)}

    def attention(self, input_tensor: torch.Tensor, image: np.ndarray,
                  alpha: float = 0.45) -> Dict[str, np.ndarray]:
        """Gradient-free attention map + overlay (requires a model)."""
        if self.model is None:
            raise ValueError("A model is required for attention maps")
        attn_gen = AttentionMapGenerator(self.model, channels_last=self.channels_last)
        try:
            attn = attn_gen.generate(input_tensor.to(self.device))
        finally:
            attn_gen.remove()
        return {"map": attn, "overlay": overlay_heatmap(image, attn, alpha=alpha)}

    def explain(
        self,
        image: np.ndarray,
        analysis: Dict[str, Any],
        input_tensor: Optional[torch.Tensor] = None,
        mask: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        output_dir: Optional[Path] = None,
        stem: str = "explanation",
    ) -> Dict[str, Any]:
        """
        Produce all available explanation artifacts for one prediction.

        Grad-CAM, attention, and confidence heatmap require both ``model`` and
        ``input_tensor``; the segmentation overlay requires ``mask``; the
        textual explanation is always produced. Images are saved to
        ``output_dir`` when provided.

        Returns:
            Dict with ``images`` (name -> np.ndarray), ``explanation`` (dict),
            and ``paths`` (name -> saved file path, when output_dir given).
        """
        images: Dict[str, np.ndarray] = {}

        if self.model is not None and input_tensor is not None:
            try:
                cam = self.gradcam(input_tensor, image, target_class)
                images["gradcam"] = cam["overlay"]
                attn = self.attention(input_tensor, image)
                images["attention"] = attn["overlay"]
                conf = analysis.get("breed_confidence") or analysis.get("species_confidence", 0.0)
                images["confidence_heatmap"] = create_confidence_heatmap(
                    image, cam["map"], conf)
            except Exception as e:  # noqa: BLE001 — explanations are best-effort
                images["error"] = np.zeros_like(image)
                analysis = {**analysis, "_explain_error": str(e)}

        if mask is not None:
            images["segmentation_overlay"] = create_segmentation_overlay(image, mask)

        explanation = generate_prediction_explanation(analysis)

        paths: Dict[str, str] = {}
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, img in images.items():
                if img is None:
                    continue
                out_path = output_dir / f"{stem}_{name}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                paths[name] = str(out_path)

        return {"images": images, "explanation": explanation, "paths": paths}
