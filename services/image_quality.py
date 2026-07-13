"""
Image Quality Assessment Service for the AI Cattle Analysis System.

Evaluates uploaded images for blur, resolution, exposure, and other
quality factors before running inference. Returns a quality score (0-100)
and specific warnings to the user.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any


class ImageQualityChecker:
    """Pre-inference image quality assessment."""

    # Thresholds (tuned for livestock photography)
    MIN_RESOLUTION = 224
    BLUR_THRESHOLD = 80.0       # Laplacian variance below this = blurry
    DARK_THRESHOLD = 50         # Mean brightness below this = too dark
    BRIGHT_THRESHOLD = 220      # Mean brightness above this = overexposed
    MIN_CONTRAST = 20           # Std deviation below this = low contrast
    MIN_ASPECT_RATIO = 0.3      # Below this = extreme aspect ratio
    MAX_ASPECT_RATIO = 3.5      # Above this = extreme aspect ratio

    def assess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess image quality and return a report.

        Args:
            image: BGR image as numpy array (from cv2.imread)

        Returns:
            Dict with:
                - quality_score: int (0-100)
                - is_acceptable: bool
                - warnings: List[str]
                - details: Dict with individual metric scores
        """
        if image is None or image.size == 0:
            return {
                "quality_score": 0,
                "is_acceptable": False,
                "warnings": ["Image could not be read or is empty."],
                "details": {},
            }

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        warnings: List[str] = []
        scores: Dict[str, float] = {}

        # 1. Resolution check
        resolution_score = self._check_resolution(h, w, warnings)
        scores["resolution"] = resolution_score

        # 2. Blur detection (Laplacian variance)
        blur_score = self._check_blur(gray, warnings)
        scores["sharpness"] = blur_score

        # 3. Brightness / Exposure
        brightness_score = self._check_brightness(gray, warnings)
        scores["brightness"] = brightness_score

        # 4. Contrast
        contrast_score = self._check_contrast(gray, warnings)
        scores["contrast"] = contrast_score

        # 5. Aspect ratio (extreme angles)
        aspect_score = self._check_aspect_ratio(h, w, warnings)
        scores["aspect_ratio"] = aspect_score

        # 6. Color saturation
        saturation_score = self._check_saturation(image, warnings)
        scores["saturation"] = saturation_score

        # Weighted overall score
        weights = {
            "resolution": 0.15,
            "sharpness": 0.30,
            "brightness": 0.20,
            "contrast": 0.15,
            "aspect_ratio": 0.10,
            "saturation": 0.10,
        }
        quality_score = sum(scores[k] * weights[k] for k in weights)
        quality_score = max(0, min(100, int(quality_score)))

        return {
            "quality_score": quality_score,
            "is_acceptable": quality_score >= 40 and len(warnings) <= 2,
            "warnings": warnings,
            "details": scores,
        }

    def _check_resolution(self, h: int, w: int, warnings: List[str]) -> float:
        """Check if image resolution is adequate."""
        min_dim = min(h, w)
        if min_dim < self.MIN_RESOLUTION:
            warnings.append(
                f"Very low resolution ({w}×{h}). Minimum recommended: {self.MIN_RESOLUTION}×{self.MIN_RESOLUTION}."
            )
            return max(0, (min_dim / self.MIN_RESOLUTION) * 50)
        elif min_dim < 448:
            warnings.append(f"Low resolution ({w}×{h}). Results may be less accurate.")
            return 60 + (min_dim - self.MIN_RESOLUTION) / (448 - self.MIN_RESOLUTION) * 30
        else:
            return min(100, 90 + (min_dim - 448) / 100)

    def _check_blur(self, gray: np.ndarray, warnings: List[str]) -> float:
        """Detect blur using Laplacian variance."""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.BLUR_THRESHOLD:
            warnings.append(
                f"Image appears blurry (sharpness: {laplacian_var:.1f}). Please use a sharper image."
            )
            return max(0, (laplacian_var / self.BLUR_THRESHOLD) * 60)
        elif laplacian_var < 200:
            return 60 + (laplacian_var - self.BLUR_THRESHOLD) / (200 - self.BLUR_THRESHOLD) * 30
        else:
            return min(100, 90 + min(10, (laplacian_var - 200) / 100))

    def _check_brightness(self, gray: np.ndarray, warnings: List[str]) -> float:
        """Check for under/over-exposure."""
        mean_brightness = np.mean(gray)
        if mean_brightness < self.DARK_THRESHOLD:
            warnings.append(
                f"Image is too dark (brightness: {mean_brightness:.0f}/255). Consider better lighting."
            )
            return max(0, (mean_brightness / self.DARK_THRESHOLD) * 50)
        elif mean_brightness > self.BRIGHT_THRESHOLD:
            warnings.append(
                f"Image is overexposed (brightness: {mean_brightness:.0f}/255)."
            )
            return max(0, ((255 - mean_brightness) / (255 - self.BRIGHT_THRESHOLD)) * 50)
        else:
            # Optimal range: center around 128
            dist_from_center = abs(mean_brightness - 128)
            return max(60, 100 - dist_from_center * 0.3)

    def _check_contrast(self, gray: np.ndarray, warnings: List[str]) -> float:
        """Check image contrast via standard deviation."""
        contrast = np.std(gray)
        if contrast < self.MIN_CONTRAST:
            warnings.append(
                f"Very low contrast ({contrast:.1f}). Image may be washed out or uniformly colored."
            )
            return max(0, (contrast / self.MIN_CONTRAST) * 50)
        elif contrast < 40:
            return 60 + (contrast - self.MIN_CONTRAST) / (40 - self.MIN_CONTRAST) * 30
        else:
            return min(100, 90 + min(10, (contrast - 40) / 10))

    def _check_aspect_ratio(self, h: int, w: int, warnings: List[str]) -> float:
        """Check for extreme aspect ratios that indicate odd angles."""
        ratio = w / max(h, 1)
        if ratio < self.MIN_ASPECT_RATIO or ratio > self.MAX_ASPECT_RATIO:
            warnings.append(
                f"Extreme aspect ratio ({ratio:.2f}:1). Image may be cropped or taken at an unusual angle."
            )
            return 40
        return 100

    def _check_saturation(self, image: np.ndarray, warnings: List[str]) -> float:
        """Check color saturation."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        if mean_saturation < 15:
            warnings.append("Image has very low color saturation (nearly grayscale).")
            return 50
        elif mean_saturation < 30:
            return 70
        else:
            return min(100, 80 + mean_saturation / 12)


# Singleton instance
_checker = ImageQualityChecker()


def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Convenience function wrapping the checker."""
    return _checker.assess(image)
