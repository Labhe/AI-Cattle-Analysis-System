import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2) # Weight and BCS
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp_cv(X, y, input_dim):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model_state = None
    best_loss = float('inf')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[val_idx], dtype=torch.float32)
        y_train, y_val = torch.tensor(y[train_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
        
        model = MLPRegressor(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        epochs = 50
        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    val_loss += criterion(pred, by).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                
    final_model = MLPRegressor(input_dim)
    final_model.load_state_dict(best_model_state)
    return final_model

def evaluate_models(models_dict, X_test, y_test):
    results = []
    
    for name, model in models_dict.items():
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        else:
            preds = model.predict(X_test)
            
        for i, target_name in enumerate(['Weight', 'BCS']):
            y_true = y_test[:, i]
            y_pred = preds[:, i]
            
            # Mask NaNs in ground truth (if any)
            mask = ~np.isnan(y_true)
            if not np.any(mask): continue
            
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            r2 = r2_score(y_true[mask], y_pred[mask])
            
            results.append({
                'Model': name,
                'Target': target_name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
            
            # Plot
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=y_true[mask], y=y_pred[mask], alpha=0.6)
            plt.plot([y_true[mask].min(), y_true[mask].max()], [y_true[mask].min(), y_true[mask].max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{name} - {target_name} Prediction')
            plt.savefig(PLOTS_DIR / f"{name.replace(' ', '_')}_{target_name}_scatter.png")
            plt.close()
            
    df = pd.DataFrame(results)
    df.to_csv(OUTPUTS_DIR / "regression_results.csv", index=False)
    print(df)
    return df

def main():
    print("Loading extracted features and annotations...")
    features_csv = OUTPUTS_DIR / "features.csv"
    annotations_csv = PROCESSED_DIR / "annotations.csv"
    
    if not features_csv.exists() or not annotations_csv.exists():
        print("Features or annotations missing. Please run previous steps.")
        return
        
    df_feat = pd.read_csv(features_csv)
    df_ann = pd.read_csv(annotations_csv)
    
    # Merge datasets
    df = pd.merge(df_feat, df_ann, on='image_path', how='inner')
    
    # Filter valid target values for training (dummy filling NaNs if 0 rows available for testing)
    df['weight_kg'].fillna(500.0, inplace=True) # Assuming some dummy fallback if mostly empty
    df['body_condition_score'].fillna(3.0, inplace=True)
    
    feature_cols = [
        'area', 'perimeter', 'bbox_length', 'bbox_width', 'aspect_ratio',
        'solidity', 'extent', 'equivalent_diameter', 'compactness',
        'convex_area', 'major_axis_length', 'minor_axis_length'
    ]
    
    target_cols = ['weight_kg', 'body_condition_score']
    
    # Split using the image splits
    with open(BASE_DIR / "data" / "splits" / "train.txt", "r") as f:
        train_paths = [l.strip() for l in f.readlines()]
    with open(BASE_DIR / "data" / "splits" / "test.txt", "r") as f:
        test_paths = [l.strip() for l in f.readlines()]
        
    train_df = df[df['image_path'].isin(train_paths)]
    test_df = df[df['image_path'].isin(test_paths)]
    
    if len(train_df) == 0:
        print("No training data matched splits for regression.")
        return
        
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_cols].values
    
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, CHECKPOINTS_DIR / "linear_reg.pkl")
    
    print("Training Random Forest...")
    rf = GridSearchCV(RandomForestRegressor(random_state=42), 
                      param_grid={'n_estimators': [50, 100], 'max_depth': [10, None]},
                      cv=5, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf.best_estimator_, CHECKPOINTS_DIR / "rf_best.pkl")
    
    print("Training XGBoost...")
    xgb = GridSearchCV(XGBRegressor(random_state=42),
                       param_grid={'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
                       cv=5, n_jobs=-1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb.best_estimator_, CHECKPOINTS_DIR / "xgb_best.pkl")
    
    print("Training MLP (PyTorch)...")
    mlp = train_mlp_cv(X_train, y_train, input_dim=len(feature_cols))
    torch.save(mlp.state_dict(), CHECKPOINTS_DIR / "mlp_best.pth")
    
    models = {
        'Linear Regression': lr,
        'Random Forest': rf.best_estimator_,
        'XGBoost': xgb.best_estimator_,
        'PyTorch MLP': mlp
    }
    
    print("Evaluating models...")
    evaluate_models(models, X_test, y_test)
    print("Regression training complete.")

if __name__ == "__main__":
    main()
