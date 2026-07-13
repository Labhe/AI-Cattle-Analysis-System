import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

def main():
    parser = argparse.ArgumentParser(description="Master Training Script")
    parser.add_argument("--model", type=str, required=True, choices=["all", "yolo", "unet", "regression", "cnn"],
                        help="Which model(s) to train.")
    
    args = parser.parse_args()
    
    pipeline = []
    if args.model in ["all", "yolo"]:
        pipeline.append(MODELS_DIR / "detection.py")
    if args.model in ["all", "unet"]:
        pipeline.append(MODELS_DIR / "segmentation.py")
    if args.model in ["all", "regression"]:
        pipeline.append(MODELS_DIR / "regression.py")
    if args.model in ["all", "cnn"]:
        pipeline.append(MODELS_DIR / "cnn_regression.py")
        
    for script in pipeline:
        print(f"--- Running {script.name} ---")
        subprocess.run(["python", str(script)], check=True)
        
    print("Training pipeline finished.")

if __name__ == "__main__":
    main()
