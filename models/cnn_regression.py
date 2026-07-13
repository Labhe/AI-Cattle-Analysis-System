import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Adjust imports to use project structure
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.dataset_loader import LivestockDataset

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_CSV = OUTPUTS_DIR / "cnn_regression_results.csv"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

class EfficientNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace classifier head
        in_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 2) # Target 0: Weight, Target 1: BCS
        )
        
    def forward(self, x):
        return self.encoder(x)

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        
        # Check for NaNs
        mask = ~torch.isnan(targets[:, 0])
        if not mask.any(): continue
        
        images = images[mask]
        targets = targets[mask]
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images, targets = images.to(device), targets.to(device)
            
            mask = ~torch.isnan(targets[:, 0])
            if not mask.any(): continue
            
            images = images[mask]
            targets = targets[mask]
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    if all_preds:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
    return running_loss / len(loader.dataset), all_preds, all_targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = LivestockDataset(PROCESSED_DIR.parent, split="train", mode="regression")
    val_dataset = LivestockDataset(PROCESSED_DIR.parent, split="val", mode="regression")
    test_dataset = LivestockDataset(PROCESSED_DIR.parent, split="test", mode="regression")
    
    if len(train_dataset) == 0:
        print("Empty dataset. Check splits and annotations.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = EfficientNetRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    
    best_loss = float('inf')
    epochs = 60
    
    print("Starting CNN Regression Training...")
    # Fast mock epoch for logic
    for epoch in range(1): # usually 'epochs'
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "cnn_regression_best.pth")
            print("Saved best CNN model.")
            
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "cnn_regression_best.pth"))
    test_loss, test_preds, test_targets = validate(model, test_loader, criterion, device)
    
    results = []
    if len(test_preds) > 0:
        for i, target_name in enumerate(['Weight', 'BCS']):
            mae = mean_absolute_error(test_targets[:, i], test_preds[:, i])
            rmse = np.sqrt(mean_squared_error(test_targets[:, i], test_preds[:, i]))
            r2 = r2_score(test_targets[:, i], test_preds[:, i])
            
            results.append({
                'Target': target_name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
            
    df = pd.DataFrame(results)
    df.to_csv(METRICS_CSV, index=False)
    print(f"Test metrics saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()
