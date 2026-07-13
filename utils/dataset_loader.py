import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class LivestockDataset(Dataset):
    def __init__(self, data_dir, split="train", mode="detection", transform=None):
        """
        Args:
            data_dir (str or Path): Path to the processed data directory.
            split (str): One of 'train', 'val', 'test'.
            mode (str): One of 'detection', 'segmentation', 'regression'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode
        
        self.images_dir = self.data_dir / "images" / split
        self.labels_dir = self.data_dir / "labels" / split
        self.masks_dir = self.data_dir / "masks" / split
        
        self.annotations_file = self.data_dir / "annotations.csv"
        if self.annotations_file.exists():
            self.df = pd.read_csv(self.annotations_file)
        else:
            self.df = None
            
        with open(self.data_dir.parent / "splits" / f"{split}.txt", "r") as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]
            
        # Default Transform (Albumentations)
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.GaussianBlur(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']) if mode == 'detection' else None)
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']) if mode == 'detection' else None)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = Path(img_path).name
        
        # Load image
        image = cv2.imread(str(self.images_dir / img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize outputs
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        bboxes = []
        class_labels = []
        weight = np.nan
        bcs = np.nan
        feature_vector = np.zeros(10, dtype=np.float32) # Placeholder for morphometric features
        
        if self.mode == "detection":
            label_file = self.labels_dir / (Path(img_name).stem + ".txt")
            if label_file.exists():
                with open(label_file, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_labels.append(int(parts[0]))
                            bboxes.append([float(x) for x in parts[1:5]])
                            
            if self.transform:
                try:
                    augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    image = augmented['image']
                    bboxes = augmented['bboxes']
                    class_labels = augmented['class_labels']
                except ValueError:
                    # In case augmentation fails (e.g. bboxes outside), fallback to just img
                    transform_no_bbox = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
                    augmented = transform_no_bbox(image=image)
                    image = augmented['image']
                    
            return image, torch.tensor(bboxes), torch.tensor(class_labels)

        elif self.mode == "segmentation":
            mask_file = self.masks_dir / img_name
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)
                
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                
            return image, mask.float().unsqueeze(0)

        elif self.mode == "regression":
            if self.df is not None:
                row = self.df[self.df['image_path'].str.contains(img_name)]
                if not row.empty:
                    weight = row.iloc[0]['weight_kg']
                    bcs = row.iloc[0]['body_condition_score']
                    
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
                
            return image, torch.tensor([weight, bcs], dtype=torch.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

def get_dataloaders(data_dir, batch_size=16, mode="detection", num_workers=4):
    train_dataset = LivestockDataset(data_dir, split="train", mode=mode)
    val_dataset = LivestockDataset(data_dir, split="val", mode=mode)
    
    # Custom collate fn for object detection to handle variable bounding boxes
    def collate_fn_det(batch):
        images, bboxes, labels = zip(*batch)
        return torch.stack(images, 0), bboxes, labels

    collate_fn = collate_fn_det if mode == "detection" else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader
