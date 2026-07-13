import os
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path()
DATA_YAML = BASE_DIR / "data" / "livestock_yolo" / "data.yaml"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs" / "detection"
METRICS_CSV = BASE_DIR / "outputs" / "detection_metrics.csv"
RUNS_DIR = BASE_DIR / "runs" / "detect"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "outputs").mkdir(parents=True, exist_ok=True)

def train_yolo():
    print("Initializing YOLOv8m model...")
    model = YOLO("yolov8m.pt")
    
    print(f"Starting training on {DATA_YAML}...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,
        patience=10,
        imgsz=640,
        project=str(RUNS_DIR),
        name="train_livestock",
        exist_ok=True
    )
    
    # Save best weights
    best_weights = RUNS_DIR / "train_livestock" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, CHECKPOINTS_DIR / "yolo_best.pt")
        print(f"Saved best weights to {CHECKPOINTS_DIR / 'yolo_best.pt'}")
        
    return model

def evaluate_yolo(model):
    print("Evaluating YOLOv8 on test set...")
    # NOTE: Ultralytics evaluates on 'val' split by default. To evaluate on test:
    # We must explicitly specify split='test' in modern versions, or modify data.yaml
    metrics = model.val(data=str(DATA_YAML), split='test', project=str(RUNS_DIR), name="val_livestock", exist_ok=True)
    
    # Extract metrics
    # metrics.results_dict contains mAP50, mAP50-95, precision, recall
    data = {
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'Fitness'],
        'Value': [
            metrics.box.map50,
            metrics.box.map,
            metrics.box.mp,
            metrics.box.mr,
            metrics.fitness
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(METRICS_CSV, index=False)
    print(f"Saved detection metrics to {METRICS_CSV}")
    
    # Predict on test images to generate annotated images
    test_images_dir = BASE_DIR / "data" / "processed" / "images" / "test"
    if test_images_dir.exists() and list(test_images_dir.glob("*.jpg")):
        print("Generating detection outputs for test set...")
        model.predict(
            source=str(test_images_dir),
            save=True,
            project=str(OUTPUTS_DIR.parent),
            name="detection",
            exist_ok=True
        )

def main():
    model = train_yolo()
    
    # If using saved model
    # model = YOLO(str(CHECKPOINTS_DIR / "yolo_best.pt"))
    evaluate_yolo(model)
    print("YOLOv8 Detection pipeline complete.")

if __name__ == "__main__":
    main()
