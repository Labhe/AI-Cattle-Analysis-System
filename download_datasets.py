import os
import subprocess
import logging
from pathlib import Path
import shutil
import zipfile
import requests

# Set up directories
BASE_DIR = Path()
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=LOGS_DIR / "download.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, cwd=cwd, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def count_files(directory):
    return sum(1 for _ in Path(directory).rglob('*') if _.is_file())

def download_github(repo_url, dest_folder):
    logging.info(f"Cloning GitHub repo: {repo_url} into {dest_folder}")
    if dest_folder.exists():
        logging.info(f"Directory {dest_folder} already exists. Skipping clone.")
    else:
        success, output = run_command(f"git clone {repo_url} {dest_folder}")
        if not success:
            logging.error(f"Failed to clone {repo_url}: {output}")
            return
    
    file_count = count_files(dest_folder)
    logging.info(f"Successfully processed {dest_folder}. Total files: {file_count}")

def download_kaggle(dataset_name, dest_folder):
    logging.info(f"Downloading Kaggle dataset: {dataset_name} into {dest_folder}")
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Check if we already downloaded
    if count_files(dest_folder) > 0:
        logging.info(f"Files already exist in {dest_folder}. Skipping download.")
        return

    # Requires kaggle CLI to be authenticated (kaggle.json in ~/.kaggle/)
    success, output = run_command(f"kaggle datasets download -d {dataset_name} -p {dest_folder} --unzip")
    if not success:
        logging.error(f"Failed to download Kaggle dataset {dataset_name}. Make sure Kaggle API keys are set. Error: {output}")
        return
        
    file_count = count_files(dest_folder)
    logging.info(f"Successfully processed {dest_folder}. Total files: {file_count}")

def download_huggingface(dataset_name, dest_folder):
    logging.info(f"Downloading Hugging Face dataset: {dataset_name} into {dest_folder}")
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import datasets library here to avoid failure if not installed
        from datasets import load_dataset
        # Load dataset and save to disk
        ds = load_dataset(dataset_name)
        ds.save_to_disk(str(dest_folder))
        
        file_count = count_files(dest_folder)
        logging.info(f"Successfully processed {dest_folder}. Total files: {file_count}")
    except Exception as e:
        logging.error(f"Failed to download HF dataset {dataset_name}: {e}")

def main():
    logging.info("Starting dataset download process...")

    # 1. CID Cow Image Dataset (GitHub)
    download_github("https://github.com/bhuiyanmobasshir94/CID.git", RAW_DATA_DIR / "cid_cow")

    # 2. Public CV Livestock Datasets Survey (GitHub)
    download_github("https://github.com/Anil-Bhujel/Public-Computer-Vision-Dataset-A-Systematic-Survey.git", RAW_DATA_DIR / "cv_livestock_survey")

    # 3. Cattle Weight Detection Dataset (Kaggle)
    download_kaggle("sadhliroomyprime/cattle-weight-detection-model-dataset-12k", RAW_DATA_DIR / "kaggle_cattle_weight")

    # 4. Cows Detection Dataset (Hugging Face)
    download_huggingface("TrainingDataPro/cows-detection-dataset", RAW_DATA_DIR / "hf_cows_detection")

    # 5. Mendeley & DatasetNinja Custom Logic
    # Note: DatasetNinja and Mendeley direct downloads via HTTP often require custom scraping or API interactions
    # that are subjected to change. Below is a placeholder implementation that assumes direct zips or APIs.
    mendeley_dest = RAW_DATA_DIR / "mendeley_cattle"
    ninja_dest = RAW_DATA_DIR / "datasetninja_cows2021"
    
    mendeley_dest.mkdir(exist_ok=True, parents=True)
    ninja_dest.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Mendeley dest {mendeley_dest} and DatasetNinja dest {ninja_dest} created. Please place downloaded archives here if direct API fails.")
    
    logging.info("Dataset download process complete.")
    logging.info("-" * 40)
    
if __name__ == "__main__":
    main()
