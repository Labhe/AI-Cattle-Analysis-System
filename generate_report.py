import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

BASE_DIR = Path()
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_evaluation_report():
    print("Generating comprehensive final report...")
    report_data = []
    
    # 1. Detection
    det_csv = OUTPUTS_DIR / "detection_metrics.csv"
    if det_csv.exists():
        df_det = pd.read_csv(det_csv)
        print("Plotting Detection metrics...")
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_det[df_det['Metric'] == 'mAP50'], x='Metric', y='Value')
        plt.title('YOLOv8 Detection mAP@0.5')
        plt.savefig(PLOTS_DIR / "detection_mAP.png")
        plt.close()
        
        for _, row in df_det.iterrows():
            report_data.append({'Task': 'Detection', 'Model': 'YOLOv8', 'Metric': row['Metric'], 'Value': row['Value']})

    # 2. Segmentation
    seg_csv = OUTPUTS_DIR / "segmentation_metrics.csv"
    if seg_csv.exists():
        df_seg = pd.read_csv(seg_csv)
        print("Plotting Segmentation metrics...")
        plt.figure(figsize=(8,6))
        sns.barplot(data=df_seg.melt(), x='variable', y='value')
        plt.title('U-Net Segmentation Metrics')
        plt.savefig(PLOTS_DIR / "segmentation_metrics.png")
        plt.close()
        
        for col in df_seg.columns:
            report_data.append({'Task': 'Segmentation', 'Model': 'U-Net', 'Metric': col, 'Value': df_seg.iloc[0][col]})

    # 3. Regression
    reg_csv = OUTPUTS_DIR / "regression_results.csv"
    if reg_csv.exists():
        df_reg = pd.read_csv(reg_csv)
        
        # Evaluate CNN Reg and concat
        cnn_csv = OUTPUTS_DIR / "cnn_regression_results.csv"
        if cnn_csv.exists():
            df_cnn = pd.read_csv(cnn_csv)
            df_cnn['Model'] = 'EfficientNet-B0'
            df_reg = pd.concat([df_reg, df_cnn])
            
        print("Plotting Regression metrics...")
        for metric in ['MAE', 'RMSE', 'R2']:
            plt.figure(figsize=(12,6))
            sns.barplot(data=df_reg, x='Model', y=metric, hue='Target')
            plt.title(f'Regression Model Comparison: {metric}')
            plt.savefig(PLOTS_DIR / f"regression_comparison_{metric}.png")
            plt.close()
            
        for _, row in df_reg.iterrows():
            for m in ['MAE', 'RMSE', 'R2']:
                report_data.append({'Task': f"Regression_{row['Target']}", 'Model': row['Model'], 'Metric': m, 'Value': row[m]})

    # Random Forest Feature Importance
    rf_path = BASE_DIR / "checkpoints" / "rf_best.pkl"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        feature_cols = [
            'area', 'perimeter', 'bbox_length', 'bbox_width', 'aspect_ratio',
            'solidity', 'extent', 'equivalent_diameter', 'compactness',
            'convex_area', 'major_axis_length', 'minor_axis_length'
        ]
        importances = rf.feature_importances_
        plt.figure(figsize=(10,8))
        sns.barplot(x=importances, y=feature_cols)
        plt.title('Random Forest Feature Importance')
        plt.savefig(PLOTS_DIR / "rf_feature_importance.png")
        plt.close()
        
    final_df = pd.DataFrame(report_data)
    final_csv = OUTPUTS_DIR / "final_report.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"Final report generated at {final_csv}")

if __name__ == "__main__":
    generate_evaluation_report()
