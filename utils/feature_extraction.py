import os
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FEATURES_CSV = OUTPUTS_DIR / "features.csv"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical body-measurement feature names used by the weight regressor
# (Phase 7). Measurements are in pixel units of the segmentation mask; with
# consistent ROI framing they correlate with live body weight, and the
# classic heart-girth formula (girth² × length) is added as a derived term.
BODY_MEASUREMENT_NAMES = [
    "body_length", "shoulder_height", "chest_width", "heart_girth", "body_area",
]


def extract_body_measurements(mask: np.ndarray) -> Dict[str, float]:
    """
    Extract cattle body measurements from a binary segmentation mask.

    Derives, in pixel units:
      - ``body_length``     : horizontal extent of the animal silhouette
      - ``shoulder_height`` : vertical extent of the front (shoulder) region
      - ``chest_width``     : vertical thickness at the chest/heart-girth column
      - ``heart_girth``     : elliptical circumference estimate around the chest
      - ``body_area``       : visible body area (mask pixel count)
      - ``heart_girth_sq_length`` : girth² × length (Schaeffer-style predictor)

    The front (head) end is taken as the side whose silhouette reaches higher,
    matching the geometry used by the Phase 4 body-part estimation.

    Args:
        mask: (H, W) or (H, W, 1) binary mask, uint8 {0, 255} or boolean.

    Returns:
        Dict of measurement name -> float. All zeros if the mask is empty.
    """
    zero = {name: 0.0 for name in BODY_MEASUREMENT_NAMES}
    zero["heart_girth_sq_length"] = 0.0

    if mask is None:
        return zero
    if mask.ndim == 3:
        mask = mask.squeeze()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return zero

    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    body_length = float(x1 - x0 + 1)
    body_area = float(binary.sum())

    # Per-column vertical thickness (rows filled) and top row per column.
    col_thickness = binary.sum(axis=0).astype(float)
    filled_cols = np.nonzero(col_thickness)[0]

    # Head end = side whose silhouette reaches higher (smaller top-row index).
    width = max(x1 - x0, 1)
    end_w = max(int(0.22 * width), 1)
    left_band = binary[:, x0:x0 + end_w]
    right_band = binary[:, max(x1 - end_w + 1, 0):x1 + 1]
    left_top = np.nonzero(left_band.any(axis=1))[0]
    right_top = np.nonzero(right_band.any(axis=1))[0]
    head_on_left = (left_top.min() if left_top.size else y1) <= \
                   (right_top.min() if right_top.size else y1)

    # Shoulder height: vertical extent in the front (shoulder) band.
    if head_on_left:
        shoulder_cols = binary[:, x0:x0 + end_w]
    else:
        shoulder_cols = binary[:, max(x1 - end_w + 1, 0):x1 + 1]
    shoulder_rows = np.nonzero(shoulder_cols.any(axis=1))[0]
    shoulder_height = float(shoulder_rows.max() - shoulder_rows.min() + 1) \
        if shoulder_rows.size else float(y1 - y0 + 1)

    # Chest column: just behind the shoulder (~1/3 from the head end).
    if head_on_left:
        chest_x = min(x0 + int(0.33 * width), binary.shape[1] - 1)
    else:
        chest_x = max(x1 - int(0.33 * width), 0)
    chest_width = float(col_thickness[chest_x])
    if chest_width == 0 and filled_cols.size:  # fall back to the nearest filled column
        nearest = filled_cols[np.argmin(np.abs(filled_cols - chest_x))]
        chest_width = float(col_thickness[nearest])

    # Heart girth: circumference of the ellipse whose axes are the chest
    # depth (vertical) and an estimated lateral width (~0.6 of the depth,
    # a standard cattle body cross-section ratio). Ramanujan approximation.
    a = chest_width / 2.0
    b = a * 0.6
    if a > 0 and b > 0:
        heart_girth = float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))
    else:
        heart_girth = 0.0

    return {
        "body_length": body_length,
        "shoulder_height": shoulder_height,
        "chest_width": chest_width,
        "heart_girth": heart_girth,
        "body_area": body_area,
        "heart_girth_sq_length": float(heart_girth ** 2 * body_length),
    }

def extract_features_from_mask(mask, target_size=(640, 640)):
    """
    Extract morphometric features from a binary mask.
    mask: np.ndarray (H, W, 1) or (H, W), boolean or uint8 (0, 255)
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
        
    mask_bool = mask > 0
    props = regionprops(mask_bool.astype(int))
    
    if len(props) == 0:
        # Return empty features if no object found
        return {
            'area': 0, 'perimeter': 0, 'bbox_length': 0, 'bbox_width': 0,
            'aspect_ratio': 0, 'solidity': 0, 'extent': 0,
            'equivalent_diameter': 0, 'compactness': 0,
            'convex_area': 0, 'major_axis_length': 0, 'minor_axis_length': 0
        }
    
    # Get the largest region if multiple
    main_region = max(props, key=lambda x: x.area)
    
    area = main_region.area
    perimeter = main_region.perimeter
    
    minr, minc, maxr, maxc = main_region.bbox
    bbox_length = maxr - minr
    bbox_width = maxc - minc
    aspect_ratio = bbox_length / (bbox_width + 1e-6)
    
    solidity = main_region.solidity
    extent = main_region.extent
    equivalent_diameter = main_region.equivalent_diameter
    
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
    
    # Catch potential math domain errors for small objects
    try:
        convex_area = main_region.convex_area
        major_axis_length = main_region.axis_major_length
        minor_axis_length = main_region.axis_minor_length
    except ValueError:
        convex_area = area
        major_axis_length = bbox_length
        minor_axis_length = bbox_width
        
    return {
        'area': area,
        'perimeter': perimeter,
        'bbox_length': bbox_length,
        'bbox_width': bbox_width,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'extent': extent,
        'equivalent_diameter': equivalent_diameter,
        'compactness': compactness,
        'convex_area': convex_area,
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length
    }

def main():
    print("Starting morphometric feature extraction...")
    masks_dir = PROCESSED_DIR / "masks"
    
    all_features = []
    
    # Iterate through train, val, test masks
    for split in ["train", "val", "test"]:
        split_dir = masks_dir / split
        if not split_dir.exists():
            continue
            
        mask_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        
        for mask_path in tqdm(mask_files, desc=f"Processing {split} masks"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            features = extract_features_from_mask(mask)
            
            # Use original image path naming convention
            img_path_str = str(PROCESSED_DIR / "images" / split / mask_path.name)
            
            feature_row = {'image_path': img_path_str}
            feature_row.update(features)
            all_features.append(feature_row)
            
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(FEATURES_CSV, index=False)
        print(f"Extracted features for {len(df)} images. Saved to {FEATURES_CSV}")
    else:
        print("No masks found for feature extraction.")

if __name__ == "__main__":
    main()
