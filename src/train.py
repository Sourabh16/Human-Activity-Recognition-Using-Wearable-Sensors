"""
train.py
Trains an SVM classifier on the preprocessed MHEALTH features.
Uses SMOTE for class balancing and GridSearchCV for hyperparameter tuning.

Authors: Sourabh Singh, Ranvir Singh
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import json
from contextlib import redirect_stdout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
CKPT_DIR = os.path.join(BASE_DIR, '..', 'checkpoints')

ACTIVITY_MAP = {
    1: "Standing still", 2: "Sitting and relaxing", 3: "Lying down",
    4: "Walking", 5: "Climbing stairs", 6: "Waist bends forward",
    7: "Frontal elevation of arms", 8: "Knees bending (crouching)",
    9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump front & back"
}


def class_names_from_encoder(le):
    """Map encoded indices back to readable activity names."""
    return [ACTIVITY_MAP.get(lbl, f'Unknown-{lbl}') for lbl in le.classes_]


def train_svm(X_train, y_train, cv=5, seed=42):
    """Apply SMOTE, run GridSearchCV, return best model + stats."""

    # balance classes with SMOTE
    counts = pd.Series(y_train).value_counts()
    min_n = counts.min()
    k = min(5, max(1, min_n - 1))
    print(f"[SMOTE] Smallest class has {min_n} samples, k_neighbors={k}")

    if min_n > 1 and k >= 1:
        try:
            sm = SMOTE(k_neighbors=k, random_state=seed)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print(f"[SMOTE] Resampled {len(y_train)} -> {len(y_res)} samples")
        except ValueError as e:
            print(f"[SMOTE] Failed ({e}), using original data")
            X_res, y_res = X_train, y_train
    else:
        print("[SMOTE] Skipped — not enough samples in minority class")
        X_res, y_res = X_train, y_train

    # grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    print("\n[GRID] Running GridSearchCV (5-fold) ...")

    svc = SVC(random_state=seed, class_weight='balanced')
    gs = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy',
                      n_jobs=-1, verbose=1)
    gs.fit(X_res, y_res)

    best = gs.best_estimator_
    train_acc = best.score(X_res, y_res)
    cv_scores = cross_val_score(best, X_res, y_res, cv=cv,
                                scoring='accuracy', n_jobs=-1)

    return best, gs.best_params_, cv_scores, train_acc


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'metrics'), exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, 'logs', 'train_output.txt')

    with open(log_path, 'w') as f:
        with redirect_stdout(f):
            print("=========================================================")
            print("       SVM MODEL TRAINING")

            try:
                X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_train_features.csv'))
                y_train = np.loadtxt(os.path.join(OUTPUT_DIR, 'y_train_labels.csv'),
                                     delimiter=',', dtype=int)
                X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_test_features.csv'))
                y_test = np.loadtxt(os.path.join(OUTPUT_DIR, 'y_test_labels.csv'),
                                    delimiter=',', dtype=int)
                with open(os.path.join(CKPT_DIR, 'label_encoder_train.pkl'), 'rb') as fp:
                    le = pickle.load(fp)
            except FileNotFoundError as e:
                print(f"[ERROR] Missing file: {e}")
                print("Run preprocessing.py first.")
                return

            X_tr = X_train.values
            X_te = X_test.values
            print(f"[DATA] Train: {X_tr.shape}, Test: {X_te.shape}")

            model, params, cv_scores, train_acc = train_svm(X_tr, y_train)
            print("\n[DONE] Training complete.")

            # evaluate on test set
            y_pred = model.predict(X_te)
            test_acc = accuracy_score(y_test, y_pred)

            names = class_names_from_encoder(le)
            report = classification_report(y_test, y_pred,
                                           target_names=names, output_dict=True)

            # save artifacts
            with open(os.path.join(CKPT_DIR, 'svm_model.pkl'), 'wb') as fp:
                pickle.dump(model, fp)
            np.savetxt(os.path.join(OUTPUT_DIR, 'y_true.csv'), y_test,
                       delimiter=',', fmt='%d')
            np.savetxt(os.path.join(OUTPUT_DIR, 'y_pred.csv'), y_pred,
                       delimiter=',', fmt='%d')
            with open(os.path.join(OUTPUT_DIR, 'metrics', 'classification_report.json'), 'w') as fp:
                json.dump(report, fp, indent=4)

            print(f"\n[SAVE] Model -> checkpoints/svm_model.pkl")
            print(f"[SAVE] Predictions -> outputs/y_true.csv, y_pred.csv")
            print(f"[SAVE] Report -> outputs/metrics/classification_report.json")

            print("\n=========================================================")
            print("            RESULTS SUMMARY")
            print("=========================================================")
            print(f"Best params: {params}")
            print(f"Train acc (resampled): {train_acc:.4f}")
            print(f"CV acc: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            print(f"Test acc (unseen subjects): {test_acc:.4f}\n")
            print(classification_report(y_test, y_pred, target_names=names))
            print("=========================================================")

    print(f"Training done. Log at {log_path}")


if __name__ == "__main__":
    main()
