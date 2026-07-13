# 🐄 Livestock Phenotyping System

**Automated Trait Estimation, Breed Recognition & Body Condition Scoring**

A deep learning-based pipeline for livestock analysis that detects animals in images, identifies their species and breed, estimates body weight, and calculates Body Condition Score (BCS) — all through a single image upload.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Models & Layer Details](#models--layer-details)
- [Datasets Used](#datasets-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Technical Details](#technical-details)

---

## Overview

Livestock Phenotyping is a multi-model inference pipeline that takes a single photograph of a farm animal and produces:

| Output | Description |
|--------|-------------|
| **Species** | Cow, Pig, Sheep, Goat, Horse, etc. |
| **Breed** | Estimated breed (e.g., Holstein Friesian, Brahman, Duroc) |
| **Scientific Name** | Latin binomial (e.g., *Bos taurus*, *Sus scrofa domesticus*) |
| **Taxonomy** | Family and Order classification |
| **Body Weight** | Estimated in kg (from both CNN and XGBoost models) |
| **Body Condition Score** | 1–5 scale (from both CNN and XGBoost models) |
| **Segmentation Mask** | Pixel-level animal body outline |
| **Confidence Score** | Detection confidence from YOLOv8 |

---

## Features

- **Multi-species detection**: Cow, Pig, Sheep, Goat, Horse, Dog, Cat, Bird, Zebra, Elephant, Bear, Giraffe
- **Breed estimation**: Color-based breed classification using ROI analysis (35+ breeds across all species)
- **Scientific classification**: Full taxonomy — scientific name, family, and order
- **Dual weight estimation**: CNN (EfficientNet-B0) and XGBoost regression models
- **Body segmentation**: U-Net with ResNet-34 encoder for pixel-level body mask
- **Web interface**: Clean, professional dashboard with drag-and-drop upload
- **Real-time inference**: Process uploaded images through the full pipeline in seconds

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                 │
│              HTML / CSS / JavaScript Frontend            │
└──────────────────────┬──────────────────────────────────┘
                       │ Upload Image
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Inference Pipeline                      │
│                                                         │
│  ┌───────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │  YOLOv8m      │──▶│  U-Net       │──▶│ Feature    │ │
│  │  Detection    │   │  Segmentation│   │ Extraction │ │
│  └───────┬───────┘   └──────────────┘   └─────┬──────┘ │
│          │                                     │        │
│          ▼                                     ▼        │
│  ┌───────────────┐              ┌──────────────────┐    │
│  │ Breed         │              │ Weight & BCS     │    │
│  │ Estimation    │              │ Regression       │    │
│  │ (Color-based) │              │ (CNN + XGBoost)  │    │
│  └───────────────┘              └──────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Models & Layer Details

### 1. YOLOv8m — Object Detection

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8 Medium (Ultralytics) |
| **Backbone** | CSPDarknet53 (Cross Stage Partial) |
| **Neck** | PANet (Path Aggregation Network) |
| **Head** | Decoupled Head (detection + classification) |
| **Parameters** | ~25.9 million |
| **Input Size** | 640 × 640 pixels |
| **Pre-trained On** | MS COCO (80 classes, 330K images) |
| **Animal Classes** | Bird (14), Cat (15), Dog (16), Horse (17), Sheep (18), Cow (19), Elephant (20), Bear (21), Zebra (22), Giraffe (23) |
| **Confidence Threshold** | 0.25 |

**Layer breakdown:**
- Input Layer: 640×640×3
- CSPDarknet Backbone: 53 convolutional layers with cross-stage partial connections
- Feature Pyramid Neck (PANet): Multi-scale feature fusion at P3, P4, P5 levels
- Decoupled Detection Head: Separate classification and regression branches
- Output: Bounding boxes + class probabilities + confidence scores

---

### 2. U-Net (ResNet-34 Encoder) — Semantic Segmentation

| Property | Value |
|----------|-------|
| **Architecture** | U-Net with ResNet-34 encoder |
| **Encoder** | ResNet-34 (5 stages, pre-trained on ImageNet) |
| **Decoder** | 5 upsampling blocks with skip connections |
| **Loss Function** | Dice Loss + Binary Cross-Entropy (DiceBCE) |
| **Input Size** | 224 × 224 × 3 (normalized) |
| **Output** | 224 × 224 × 1 (binary mask) |
| **Pre-trained On** | ImageNet (encoder only) |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=80) |

**Layer breakdown:**

| Stage | Layers | Channels | Description |
|-------|--------|----------|-------------|
| Encoder 0 | Conv2d → BN → ReLU | 3 → 64 | Initial convolution |
| Encoder 1 | MaxPool → ResBlock ×3 | 64 → 64 | Stage 1 residual blocks |
| Encoder 2 | ResBlock ×4 | 64 → 128 | Stage 2 residual blocks |
| Encoder 3 | ResBlock ×6 | 128 → 256 | Stage 3 residual blocks |
| Encoder 4 | ResBlock ×3 | 256 → 512 | Stage 4 residual blocks |
| Decoder 4 | ConvTranspose2d → Conv ×2 | 512 → 256 | Up + skip from E3 |
| Decoder 3 | ConvTranspose2d → Conv ×2 | 256 → 128 | Up + skip from E2 |
| Decoder 2 | ConvTranspose2d → Conv ×2 | 128 → 64 | Up + skip from E1 |
| Decoder 1 | ConvTranspose2d → Conv ×2 | 64 → 64 | Up + skip from E0 |
| Decoder 0 | ConvTranspose2d → Conv ×2 | 64 → 32 | Final upsampling |
| Output | Conv2d 1×1 | 32 → 1 | Binary segmentation mask |

**Total encoder parameters**: ~21.3 million (ResNet-34)
**Total decoder parameters**: ~4.2 million
**Total model parameters**: ~25.5 million

---

### 3. EfficientNet-B0 — CNN Weight & BCS Regression

| Property | Value |
|----------|-------|
| **Architecture** | EfficientNet-B0 (modified head) |
| **Backbone** | MBConv blocks with squeeze-excitation |
| **Parameters** | ~5.3 million |
| **Input Size** | 224 × 224 × 3 (ImageNet normalized) |
| **Output** | 2 values (Weight in kg, BCS 1-5 scale) |
| **Pre-trained On** | ImageNet-1K |
| **Dropout** | 0.4 |
| **Loss** | MSE Loss |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=60) |

**Layer breakdown:**
- Stem: Conv2d 3×3 (3 → 32 channels)
- Stage 1: MBConv1 ×1 (32 → 16)
- Stage 2: MBConv6 ×2 (16 → 24)
- Stage 3: MBConv6 ×2 (24 → 40)
- Stage 4: MBConv6 ×3 (40 → 80)
- Stage 5: MBConv6 ×3 (80 → 112)
- Stage 6: MBConv6 ×4 (112 → 192)
- Stage 7: MBConv6 ×1 (192 → 320)
- Head: Conv2d 1×1 (320 → 1280) → AdaptiveAvgPool → Dropout(0.4) → Linear(1280 → 2)

---

### 4. XGBoost — Feature-Based Weight & BCS Regression

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost (Gradient Boosted Trees) |
| **Input Features** | 12 morphometric features extracted from segmentation mask |
| **Output** | 2 values (Weight in kg, BCS 1-5 scale) |
| **Framework** | scikit-learn compatible (via `xgboost` package) |

**Input features extracted from segmentation mask:**

| # | Feature | Description |
|---|---------|-------------|
| 1 | `area` | Pixel area of segmented region |
| 2 | `perimeter` | Boundary perimeter in pixels |
| 3 | `bbox_length` | Bounding box height |
| 4 | `bbox_width` | Bounding box width |
| 5 | `aspect_ratio` | Width / Height ratio |
| 6 | `solidity` | Area / Convex Hull Area |
| 7 | `extent` | Area / Bounding Box Area |
| 8 | `equivalent_diameter` | Diameter of circle with same area |
| 9 | `compactness` | 4π × Area / Perimeter² |
| 10 | `convex_area` | Area of convex hull |
| 11 | `major_axis_length` | Length of fitted ellipse major axis |
| 12 | `minor_axis_length` | Length of fitted ellipse minor axis |

---

### 5. Breed Estimation — Color-Based Classifier

| Property | Value |
|----------|-------|
| **Method** | ROI color histogram analysis |
| **Profiles** | 8 color types (white, dark, black_white, light_brown, light_gray, red_brown, red_white, brown) |
| **Breeds Covered** | 35+ breeds across 13 species |
| **Features Analyzed** | Avg R/G/B channels, brightness, color variance (std) |

**Breeds per species:**

| Species | Scientific Name | Number of Breeds | Examples |
|---------|----------------|-----------------|----------|
| Cow | *Bos taurus* | 7 | Holstein Friesian, Jersey, Angus, Brahman, Charolais |
| Pig | *Sus scrofa domesticus* | 5 | Large White, Landrace, Duroc, Berkshire, Hampshire |
| Sheep | *Ovis aries* | 5 | Merino, Suffolk, Dorper, Romney, Corriedale |
| Goat | *Capra aegagrus hircus* | 5 | Boer, Saanen, Alpine, Nubian, Jamunapari |
| Horse | *Equus caballus* | 5 | Thoroughbred, Arabian, Quarter Horse, Clydesdale, Appaloosa |

---

## Datasets Used

### MS COCO (Common Objects in Context)

| Property | Value |
|----------|-------|
| **Source** | [cocodataset.org](https://cocodataset.org) |
| **Used For** | YOLOv8 pre-training (object detection) |
| **Size** | 330,000+ images, 2.5M labeled instances |
| **Animal Classes** | 10 (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe) |
| **Format** | COCO JSON annotations |

### ImageNet-1K (ILSVRC 2012)

| Property | Value |
|----------|-------|
| **Source** | [image-net.org](https://www.image-net.org) |
| **Used For** | Pre-training ResNet-34 (segmentation encoder) and EfficientNet-B0 (regression backbone) |
| **Size** | 1.28M training images, 1000 classes |
| **Relevance** | Transfer learning — features from ImageNet generalize well to animal body shapes |

### CIFAR-100

| Property | Value |
|----------|-------|
| **Source** | [HuggingFace](https://huggingface.co/datasets/cifar100) |
| **Used For** | Fine-tuning YOLOv8 on livestock-specific classes |
| **Size** | 60,000 images (32×32), 100 fine-grained classes |
| **Relevant Classes** | cattle |

---

## Project Structure

```
livestock_phenotyping/
├── app.py                        # Flask web server
├── inference.py                   # Main inference pipeline (all models)
├── train.py                       # Training orchestrator
├── train_yolo.py                  # Standalone YOLO fine-tuning script
├── download_livestock_dataset.py  # Dataset downloader (HuggingFace + Roboflow)
├── download_datasets.py           # General dataset download utility
├── prepare_dataset.py             # Data preprocessing & splits
├── generate_report.py             # Results report generator
├── test.py                        # Test script
├── config.yaml                    # Project configuration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── models/
│   ├── detection.py               # YOLOv8 training script
│   ├── segmentation.py            # U-Net (ResNet-34) training script
│   ├── cnn_regression.py          # EfficientNet-B0 regression training
│   └── regression.py              # XGBoost regression training
│
├── utils/
│   ├── dataset_loader.py          # PyTorch Dataset class for all modes
│   └── feature_extraction.py      # Morphometric feature extractor
│
├── static/
│   ├── style.css                  # Frontend styles
│   └── script.js                  # Frontend logic
│
├── templates/
│   └── index.html                 # Dashboard HTML
│
├── checkpoints/                   # Saved model weights
│   └── yolo_best.pt               # Fine-tuned YOLOv8 weights
│
├── data/
│   ├── data.yaml                  # Dataset configuration
│   └── livestock_yolo/            # YOLO format dataset
│       ├── data.yaml
│       ├── train/
│       ├── valid/
│       └── test/
│
├── outputs/
│   └── inference/                 # Annotated output images & results CSV
│
└── runs/                          # YOLO training logs & tensorboard
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4 GB RAM minimum (8 GB recommended)

### Step-by-step

```bash
# 1. Navigate to the project directory
cd livestock_phenotyping

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows (PowerShell):
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Download & fine-tune YOLO on livestock data
python download_livestock_dataset.py
python train_yolo.py
```

---

## How to Run

```bash
# Activate virtual environment
.\venv\Scripts\activate   # Windows
source venv/bin/activate   # macOS/Linux

# Start the web application
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

1. Click the upload area or drag & drop an animal image
2. Click **Process Image**
3. View results: species, breed, scientific name, weight, BCS, and annotated image

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the web dashboard |
| POST | `/upload` | Upload image → returns JSON with all predictions |
| GET | `/outputs/<filename>` | Serves annotated output images |

### POST `/upload` Response Example

```json
{
  "success": true,
  "results": {
    "image": "cow_photo.jpg",
    "species": "Cow",
    "breed": "Brahman",
    "scientific_name": "Bos taurus",
    "family": "Bovidae",
    "order": "Artiodactyla",
    "weight_range": "500-900 kg",
    "class_id": 19,
    "confidence": 92.3,
    "xgb_weight_kg": 612.45,
    "xgb_bcs": 3.21,
    "cnn_weight_kg": 608.92,
    "cnn_bcs": 3.18
  },
  "annotated_image": "cow_photo_annotated.jpg"
}
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep learning framework |
| `torchvision` | ≥0.15 | Pre-trained models (ResNet-34, EfficientNet-B0) |
| `ultralytics` | ≥8.0 | YOLOv8 object detection |
| `opencv-python` | ≥4.8 | Image processing |
| `scikit-learn` | ≥1.3 | ML utilities |
| `xgboost` | ≥1.7 | Gradient boosted regression |
| `albumentations` | ≥1.3 | Data augmentation |
| `pandas` | ≥2.0 | Data handling |
| `numpy` | ≥1.24 | Numerical computing |
| `flask` | ≥2.3 | Web server |
| `datasets` | ≥2.14 | HuggingFace dataset loading |
| `joblib` | ≥1.3 | Model serialization |
| `tqdm` | ≥4.65 | Progress bars |

---

## Technical Details

### Inference Pipeline Flow

1. **Image Upload** → Flask receives the image via POST
2. **Object Detection** → YOLOv8m detects all objects, filters for animal classes (COCO IDs 14–23)
3. **Species Mapping** → COCO class ID is mapped to species name (e.g., 19 → Cow)
4. **Secondary Classification** → Color analysis checks for pig (pink skin tone) or goat (not in COCO)
5. **Breed Estimation** → ROI color histogram analysis matches to known breed color profiles
6. **ROI Extraction** → Crop the detected animal from the image
7. **Segmentation** → U-Net produces a binary body mask
8. **Feature Extraction** → 12 morphometric features from the mask (area, perimeter, solidity, etc.)
9. **XGBoost Regression** → Predicts weight & BCS from morphometric features
10. **CNN Regression** → EfficientNet-B0 predicts weight & BCS from the ROI pixels
11. **Visualization** → Annotated image with bounding box, mask overlay, and labels
12. **Response** → JSON with all predictions + annotated image URL

### Key Design Decisions

- **COCO Pre-trained YOLOv8m** is used for detection instead of fine-tuned models because COCO has 330K images with professional annotations — far superior to small custom datasets
- **Color-based breed estimation** analyzes the center 60% of the animal ROI to determine dominant color profile, then matches against a database of known breed color patterns
- **Dual regression** (CNN + XGBoost) provides two independent weight estimates for cross-validation
- **Fallback mechanisms** at every stage ensure the pipeline never fails — if YOLO finds no animal, the full image is used; if segmentation fails, a synthetic mask is generated

---

## License

This project is for educational and research purposes.

---

*Built with PyTorch, Ultralytics YOLOv8, Flask, and OpenCV*
