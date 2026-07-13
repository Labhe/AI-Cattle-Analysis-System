import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

def main():
    parser = argparse.ArgumentParser(description="Master Testing Script")
    parser.add_argument("--model", type=str, default="all", help="Which model to test (currently ignored, runs all locally).")
    
    args = parser.parse_args()
    
    # In a full setup, test.py would hook into model evaluation fns specifically.
    # Here, our models evaluate internally during script run.
    # We will trigger the evaluation/report generation directly.
    print(f"--- Running Final Evaluation Report Generation ---")
    script = BASE_DIR / "generate_report.py"
    subprocess.run(["python", str(script)], check=True)
    
    print("Testing pipeline finished. See outputs/ for results.")

if __name__ == "__main__":
    main()
