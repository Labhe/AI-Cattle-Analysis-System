import os
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
