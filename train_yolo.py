"""
Directly train YOLOv8 on the livestock dataset and save weights.
Run this standalone - no subprocess wrapper needed.
"""
from pathlib import Path
from ultralytics import YOLO
import shutil

BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "data" / "livestock_yolo" / "data.yaml"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 50)
    print("Fine-tuning YOLOv8m on Livestock Dataset")
    print("=" * 50)
    
    model = YOLO("yolov8m.pt")
    
    print(f"Training on {DATA_YAML}...")
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=30,
            patience=10,
            imgsz=640,
            batch=4,
            workers=0,  # Windows compatibility
            project=str(BASE_DIR / "runs" / "detect"),
            name="train_livestock",
            exist_ok=True,
            verbose=True
        )
    except Exception as e:
        print(f"Training error: {e}")
        print("Attempting with fewer epochs...")
        results = model.train(
            data=str(DATA_YAML),
            epochs=10,
            patience=5,
            imgsz=320,
            batch=2,
            workers=0,
            project=str(BASE_DIR / "runs" / "detect"),
            name="train_livestock",
            exist_ok=True,
            verbose=True
        )
    
    # Copy best weights
    best_weights = BASE_DIR / "runs" / "detect" / "train_livestock" / "weights" / "best.pt"
    last_weights = BASE_DIR / "runs" / "detect" / "train_livestock" / "weights" / "last.pt"
    
    if best_weights.exists():
        shutil.copy(best_weights, CHECKPOINTS_DIR / "yolo_best.pt")
        print(f"Saved best weights to {CHECKPOINTS_DIR / 'yolo_best.pt'}")
    elif last_weights.exists():
        shutil.copy(last_weights, CHECKPOINTS_DIR / "yolo_best.pt")
        print(f"Saved last weights to {CHECKPOINTS_DIR / 'yolo_best.pt'}")
    else:
        print("WARNING: No trained weights found!")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
