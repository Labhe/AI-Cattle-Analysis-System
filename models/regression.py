import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


# ════════════════════════════════════════════════════════════════════════
#  Regression metrics (shared by weight & BCS regressors — Phases 7 & 8)
# ════════════════════════════════════════════════════════════════════════


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE (%), and R² for a set of predictions.

    Args:
        y_true: Ground-truth targets, shape (N,).
        y_pred: Predicted values, shape (N,).

    Returns:
        Dict with ``mae``, ``rmse``, ``mape``, ``r2`` (NaN-safe; empty input
        yields NaNs).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "mape": float("nan"), "r2": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    denom = np.where(np.abs(yt) < 1e-6, 1e-6, yt)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape": mape,
        "r2": r2,
    }


def build_tree_regressors(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Instantiate the tree-based regressors named in the config, skipping any
    whose library is not installed.

    Supports ``xgboost``, ``catboost``, ``lightgbm``, and ``random_forest``.
    Hyperparameters are read from ``weight_regressor.training.<name>`` when
    present. Returns an ordered mapping of display name -> estimator.
    """
    config = config or {}
    train_cfg = config.get("weight_regressor", {}).get("training", {})
    wanted = train_cfg.get("regressors_to_compare",
                           ["xgboost", "catboost", "lightgbm", "random_forest"])
    seed = int(train_cfg.get("seed", 42))
    regressors: Dict[str, Any] = {}

    for name in wanted:
        try:
            if name == "xgboost":
                p = train_cfg.get("xgboost", {})
                regressors["XGBoost"] = XGBRegressor(
                    n_estimators=int(p.get("n_estimators", 200)),
                    max_depth=int(p.get("max_depth", 8)),
                    learning_rate=float(p.get("learning_rate", 0.05)),
                    random_state=seed, n_jobs=-1)
            elif name == "catboost":
                from catboost import CatBoostRegressor
                p = train_cfg.get("catboost", {})
                regressors["CatBoost"] = CatBoostRegressor(
                    iterations=int(p.get("iterations", 200)),
                    depth=int(p.get("depth", 8)),
                    learning_rate=float(p.get("learning_rate", 0.05)),
                    random_state=seed, verbose=False)
            elif name == "lightgbm":
                from lightgbm import LGBMRegressor
                p = train_cfg.get("lightgbm", {})
                regressors["LightGBM"] = LGBMRegressor(
                    n_estimators=int(p.get("n_estimators", 200)),
                    max_depth=int(p.get("max_depth", 8)),
                    learning_rate=float(p.get("learning_rate", 0.05)),
                    random_state=seed, n_jobs=-1, verbose=-1)
            elif name == "random_forest":
                p = train_cfg.get("random_forest", {})
                regressors["Random Forest"] = RandomForestRegressor(
                    n_estimators=int(p.get("n_estimators", 200)),
                    max_depth=p.get("max_depth", None),
                    random_state=seed, n_jobs=-1)
        except ImportError:
            # Optional dependency missing — skip this candidate.
            continue
    return regressors


@dataclass
class WeightRegressor:
    """
    Body-measurement → live-weight regressor (Phase 7).

    Compares several tree-based regressors, selects the best by validation
    RMSE, and predicts weight with a confidence score and a prediction
    interval derived from the validation residual spread.

    Attributes:
        feature_names: Ordered feature names the model expects.
        model: The selected fitted estimator.
        best_model_name: Display name of the selected model.
        metrics: Per-model validation metric dicts.
        residual_std: Std of validation residuals (drives the interval).
        r2: Validation R² of the selected model (drives the confidence score).
    """

    feature_names: List[str]
    model: Optional[Any] = None
    best_model_name: Optional[str] = None
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    residual_std: float = 0.0
    r2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, config: Optional[Dict[str, Any]] = None,
            val_fraction: float = 0.2, seed: int = 42) -> "WeightRegressor":
        """
        Train and compare candidate regressors; keep the best (lowest RMSE).

        Args:
            X: Feature matrix (N, F) aligned with :attr:`feature_names`.
            y: Weight targets (N,).
            config: Parsed ``model_config.yaml`` dict for hyperparameters.
            val_fraction: Held-out fraction for model selection.
            seed: RNG seed for the split.

        Raises:
            ValueError: If there are too few samples or no candidate models.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) < 5:
            raise ValueError(f"Need at least 5 samples to train, got {len(X)}")

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        candidates = build_tree_regressors(config)
        if not candidates:
            raise ValueError("No regression candidates available (check installed libraries)")

        best_rmse = float("inf")
        for name, estimator in candidates.items():
            estimator.fit(X_tr, y_tr)
            preds = estimator.predict(X_val)
            m = regression_metrics(y_val, preds)
            self.metrics[name] = m
            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                self.model = estimator
                self.best_model_name = name
                residuals = y_val - preds
                self.residual_std = float(np.std(residuals)) if len(residuals) > 1 else best_rmse
                self.r2 = m["r2"] if not np.isnan(m["r2"]) else 0.0

        # Refit the winner on all data for the deployed model.
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict weight(s) for a feature matrix (N, F)."""
        if self.model is None:
            raise RuntimeError("WeightRegressor is not fitted")
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float)

    def predict_with_interval(self, X: np.ndarray, z: float = 1.96) -> Dict[str, Any]:
        """
        Predict weight with a confidence score and prediction interval for a
        single sample.

        Args:
            X: Feature matrix (1, F) or (F,).
            z: Std multiplier for the interval (1.96 ≈ 95%).

        Returns:
            Dict with ``weight_kg``, ``confidence`` (0–100), ``interval``
            [lower, upper], ``interval_kg`` half-width, and ``model``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        weight = float(self.predict(X)[0])
        half = z * self.residual_std
        # Confidence blends fit quality (R²) with relative interval tightness.
        rel = half / weight if weight > 1e-6 else 1.0
        confidence = max(0.0, min(100.0, 100.0 * (0.5 * max(self.r2, 0.0)
                                                   + 0.5 * max(0.0, 1.0 - rel))))
        return {
            "weight_kg": round(weight, 1),
            "confidence": round(confidence, 1),
            "interval": [round(weight - half, 1), round(weight + half, 1)],
            "interval_kg": round(half, 1),
            "model": self.best_model_name,
        }

    def save(self, path: "os.PathLike") -> Path:
        """Persist the full regressor (model + metadata) via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_names": self.feature_names,
            "model": self.model,
            "best_model_name": self.best_model_name,
            "metrics": self.metrics,
            "residual_std": self.residual_std,
            "r2": self.r2,
        }, path)
        return path

    @classmethod
    def load(cls, path: "os.PathLike") -> "WeightRegressor":
        """Load a regressor previously written by :meth:`save`."""
        payload = joblib.load(path)
        if isinstance(payload, dict) and "feature_names" in payload:
            return cls(
                feature_names=payload["feature_names"],
                model=payload.get("model"),
                best_model_name=payload.get("best_model_name"),
                metrics=payload.get("metrics", {}),
                residual_std=payload.get("residual_std", 0.0),
                r2=payload.get("r2", 0.0),
            )
        # Backward compatibility: a bare estimator pickle (legacy checkpoints).
        reg = cls(feature_names=[])
        reg.model = payload
        reg.best_model_name = type(payload).__name__
        return reg

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
