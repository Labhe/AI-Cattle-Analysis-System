import os
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import imagehash
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil

# Configuration
BASE_DIR = Path()
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"
OUTPUTS_DIR = BASE_DIR / "outputs" / "dataset_stats"

for d in [PROCESSED_DIR / "images", PROCESSED_DIR / "labels", PROCESSED_DIR / "masks", SPLITS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    
for split in ["train", "val", "test"]:
    (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "masks" / split).mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {
    'cow': 0, 'cattle': 0, 'cows': 0,
    'pig': 1, 'pigs': 1,
    'goat': 2, 'goats': 2,
    'sheep': 3,
    'horse': 4, 'horses': 4,
    'other': 5
}

TARGET_SIZE = (640, 640)

def get_class_id(name):
    name = name.lower().strip()
    return CLASS_MAPPING.get(name, 5)

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def hash_image(filepath):
    try:
        with Image.open(filepath) as img:
            return imagehash.phash(img)
    except Exception:
        return None

def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        class_id = get_class_id(name)
        
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)
        
        # YOLO format
        dw = 1. / img_width
        dh = 1. / img_height
        x = (xmin + xmax) / 2.0 - 1
        y = (ymin + ymax) / 2.0 - 1
        w = xmax - xmin
        h = ymax - ymin
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        yolo_annotations.append(f"{class_id} {x} {y} {w} {h}")
    return yolo_annotations

def generate_stats_and_splits(records):
    df = pd.DataFrame(records)
    
    # Save master annotations
    df.to_csv(PROCESSED_DIR / "annotations.csv", index=False)
    
    # Stratified split if possible (assumes enough class samples)
    if len(df) > 0:
        # We simulate a target class list
        class_ids = [CLASS_MAPPING.get(cls, 5) for cls in df['animal_class']]
        try:
            train_df, tmp_df = train_test_split(df, test_size=0.3, stratify=class_ids, random_state=42)
            val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42) # Not stratified on tmp because might be too small
        except ValueError:
            # Fallback if stratify fails
            train_df, tmp_df = train_test_split(df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
        
        def save_split_and_move(split_df, split_name):
            split_txt = SPLITS_DIR / f"{split_name}.txt"
            with open(split_txt, 'w') as f:
                for _, row in split_df.iterrows():
                    img_path = str(row['image_path'])
                    f.write(img_path + "\n")
                    
                    # Also move the file to the right subdirectory in processed if it was in the root
                    img_name = Path(img_path).name
                    dest_path = PROCESSED_DIR / "images" / split_name / img_name
                    
                    if Path(img_path).exists() and img_path != str(dest_path):
                        shutil.move(img_path, dest_path)
                        # Also move labels
                        label_name = img_name.rsplit('.', 1)[0] + '.txt'
                        label_src = PROCESSED_DIR / "labels" / label_name
                        label_dest = PROCESSED_DIR / "labels" / split_name / label_name
                        if label_src.exists():
                            shutil.move(label_src, label_dest)
        
        save_split_and_move(train_df, "train")
        save_split_and_move(val_df, "val")
        save_split_and_move(test_df, "test")
        
        # Plot Distributions
        plt.figure(figsize=(10,6))
        sns.countplot(data=df, x='animal_class')
        plt.title('Distribution of Animal Classes')
        plt.savefig(OUTPUTS_DIR / "class_distribution.png")
        plt.close()
        
        print(f"Total dataset size: {len(df)}")
        print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

def main():
    print("Starting dataset preparation...")
    
    seen_hashes = set()
    records = []
    processed_count = 0
    
    image_files = list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.jpeg")) + list(RAW_DIR.rglob("*.png"))
    print(f"Found {len(image_files)} potential image files across raw datasets.")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        if not is_valid_image(img_path):
            continue
            
        img_hash = hash_image(img_path)
        if img_hash is None or img_hash in seen_hashes:
            continue
        
        seen_hashes.add(img_hash)
        
        # Read and resize
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, TARGET_SIZE)
        
        new_filename = f"{img_path.stem}_{processed_count}.jpg"
        new_filepath = PROCESSED_DIR / "images" / new_filename
        cv2.imwrite(str(new_filepath), img_resized)
        
        # Default annotations (can be parsed from sidecars)
        animal_class = 'cow'  # heuristic default
        source_dataset = img_path.parent.name
        
        # Attempt to find bounding boxes
        xml_path = img_path.with_suffix('.xml')
        txt_path = img_path.with_suffix('.txt')
        
        label_filepath = PROCESSED_DIR / "labels" / f"{new_filepath.stem}.txt"
        
        if xml_path.exists():
            yolo_lines = convert_voc_to_yolo(xml_path, w, h)
            with open(label_filepath, 'w') as f:
                f.write('\n'.join(yolo_lines))
        elif txt_path.exists():
            # Assume already YOLO format, just copy (ensure class ids are mapped properly if needed)
            shutil.copy(txt_path, label_filepath)
        else:
            # Create dummy full-image bounding box if no label exists (assuming classification dataset)
            class_id = get_class_id(animal_class)
            with open(label_filepath, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")
        
        records.append({
            'image_path': str(new_filepath),
            'animal_class': animal_class,
            'weight_kg': np.nan,
            'body_condition_score': np.nan,
            'length_px': w,
            'width_px': h,
            'source_dataset': source_dataset
        })
        
        processed_count += 1
        
    generate_stats_and_splits(records)
    
    # Generate data.yaml
    yaml_config = {
        'path': str(PROCESSED_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_MAPPING),
        'names': {v: k for k, v in CLASS_MAPPING.items() if v != 5 and v == list(CLASS_MAPPING.values()).index(v)} 
        # Hacky inverted map
    }
    # Better class names config
    names_dict = {0: 'cow', 1: 'pig', 2: 'goat', 3: 'sheep', 4: 'horse', 5: 'other'}
    yaml_config['names'] = names_dict
    
    with open(BASE_DIR / 'data' / 'data.yaml', 'w') as f:
        yaml.dump(yaml_config, f, sort_keys=False)
        
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
