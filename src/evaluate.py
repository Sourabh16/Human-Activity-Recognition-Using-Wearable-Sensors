"""
evaluate.py
Loads predictions from train.py and generates evaluation plots:
  - Normalised confusion matrix heatmap
  - PCA-based sensor correlation scatter plot

Authors: Sourabh Singh, Ranvir Singh
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from contextlib import redirect_stdout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
CKPT_DIR = os.path.join(BASE_DIR, '..', 'checkpoints')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

ACTIVITY_MAP = {
    1: "Standing still", 2: "Sitting and relaxing", 3: "Lying down",
    4: "Walking", 5: "Climbing stairs", 6: "Waist bends forward",
    7: "Frontal elevation of arms", 8: "Knees bending (crouching)",
    9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump front & back"
}


def class_names_from_encoder(le):
    return [ACTIVITY_MAP.get(l, f'Unknown-{l}') for l in le.classes_]


def plot_confusion(y_true, y_pred, names, save_path):
    """Normalised confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linecolor='gray', linewidths=0.5)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.ylabel('True Activity', fontsize=14)
    plt.xlabel('Predicted Activity', fontsize=14)
    plt.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved confusion matrix -> {save_path}")


def plot_pca_correlation(X_df, target_prefix, save_path):
    """
    Reduce target-sensor features and other-sensor features each to PC1,
    then scatter-plot their correlation with a regression line.
    """
    target_cols = [c for c in X_df.columns if c.startswith(target_prefix)]
    other_cols  = [c for c in X_df.columns if not c.startswith(target_prefix)]

    if not target_cols or not other_cols:
        print(f"[PLOT] Skipping correlation — missing columns for '{target_prefix}'")
        return

    pc_target = PCA(n_components=1).fit_transform(X_df[target_cols].values).flatten()
    pc_other  = PCA(n_components=1).fit_transform(X_df[other_cols].values).flatten()
    r = np.corrcoef(pc_target, pc_other)[0, 1]

    pca_t = PCA(n_components=1).fit(X_df[target_cols].values)
    pca_o = PCA(n_components=1).fit(X_df[other_cols].values)

    plt.figure(figsize=(10, 8))
    sns.regplot(x=pc_target, y=pc_other,
                scatter_kws={'alpha': 0.4, 's': 15},
                line_kws={"color": "red", "lw": 2})

    name = target_prefix.replace('_', '').title()
    plt.title(f'Correlation: {name} PC1 vs Other Sensors PC1  (R = {r:.3f})',
              fontsize=14)
    plt.xlabel(f'{name} PC1 ({pca_t.explained_variance_ratio_[0]*100:.1f}% var)',
               fontsize=12)
    plt.ylabel(f'Other Sensors PC1 ({pca_o.explained_variance_ratio_[0]*100:.1f}% var)',
               fontsize=12)
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved correlation plot -> {save_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, 'logs', 'evaluation_output.txt')

    with open(log_path, 'w') as f:
        with redirect_stdout(f):
            print("=========================================================")
            print("       MODEL EVALUATION & VISUALIZATION")
            print("=========================================================")

            try:
                y_true = np.loadtxt(os.path.join(OUTPUT_DIR, 'y_true.csv'),
                                    delimiter=',', dtype=int)
                y_pred = np.loadtxt(os.path.join(OUTPUT_DIR, 'y_pred.csv'),
                                    delimiter=',', dtype=int)
                with open(os.path.join(CKPT_DIR, 'label_encoder_train.pkl'), 'rb') as fp:
                    le = pickle.load(fp)
                X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_test_features.csv'))
            except FileNotFoundError as e:
                print(f"[ERROR] {e} — run train.py first.")
                return

            names = class_names_from_encoder(le)
            print(f"[DATA] Loaded {len(names)} classes, {len(y_true)} predictions")

            plot_confusion(y_true, y_pred, names,
                           os.path.join(FIG_DIR, 'confusion_matrix.png'))
            plot_pca_correlation(X_test, 'ankle_',
                                os.path.join(FIG_DIR, 'reduced_ankle_vs_others_correlation.png'))

            print("\n[DONE] All plots saved to outputs/figures/")
            print("=========================================================")

    print(f"Evaluation done. Log at {log_path}")


if __name__ == "__main__":
    main()
