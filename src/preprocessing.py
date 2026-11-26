"""
preprocessing.py
Loads the MHEALTH dataset, cleans it, extracts time/freq domain features
from sliding windows, and saves train/test splits for the SVM pipeline.

Authors: Sourabh Singh, Ranvir Singh
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from contextlib import redirect_stdout
import warnings

warnings.filterwarnings("ignore")

# --- paths ---
# set MHEALTH_DATA_DIR env var if your data lives elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    'MHEALTH_DATA_DIR',
    os.path.join(BASE_DIR, '..', 'dataset', 'MHEALTHDATASET')
)
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
CKPT_DIR = os.path.join(BASE_DIR, '..', 'checkpoints')

# sensor column names matching the .log files (24 cols total)
COL_NAMES = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'ecg_lead1', 'ecg_lead2',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
    'label'
]

ACTIVITY_MAP = {
    1: "Standing still", 2: "Sitting and relaxing", 3: "Lying down",
    4: "Walking", 5: "Climbing stairs", 6: "Waist bends forward",
    7: "Frontal elevation of arms", 8: "Knees bending (crouching)",
    9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump front & back"
}

FS = 50        # sampling rate in Hz
WIN_LEN = 256  # window size (256 samples = 5.12s)
STEP = 128     # step size (50% overlap)


def load_data(data_dir, subjects):
    """Read .log files for given subject IDs, concat into one DataFrame."""
    frames = []
    print(f"[LOAD] Loading subjects: {subjects} ...")
    for s in subjects:
        fpath = os.path.join(data_dir, f'mHealth_subject{s}.log')
        df = pd.read_csv(fpath, sep=r'\s+', header=None, names=COL_NAMES)
        df['subject'] = s
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[LOAD] Got {len(combined):,} raw samples.")
    return combined


def clean(df, drop_null=True, drop_ecg=True):
    """Remove null-class rows, drop ECG cols, forward-fill NaNs."""
    out = df.copy()

    if drop_null:
        before = len(out)
        out = out[out['label'] != 0].reset_index(drop=True)
        print(f"[CLEAN] Dropped {before - len(out):,} null-class rows.")

    if drop_ecg:
        out.drop(columns=['ecg_lead1', 'ecg_lead2'], errors='ignore', inplace=True)

    nans = out.isnull().sum().sum()
    if nans > 0:
        out.fillna(method='ffill', inplace=True)
        print(f"[CLEAN] Forward-filled {nans:,} NaN values.")

    return out


def scale_and_encode(df, le=None, scaler=None, fit=True):
    """
    Z-score standardise sensor columns; label-encode the activity column.
    When fit=True, fits new scaler/encoder on this data (training).
    When fit=False, transforms using already-fitted objects (test).
    """
    out = df.copy()
    sensor_cols = [c for c in out.columns if c not in ('label', 'subject')]

    if fit:
        le = LabelEncoder()
        out['label'] = le.fit_transform(out['label'])
        scaler = StandardScaler()
        out[sensor_cols] = scaler.fit_transform(out[sensor_cols])
    else:
        # only keep labels the encoder has seen
        mask = out['label'].isin(le.classes_)
        out = out[mask].copy()
        out['label'] = le.transform(out['label'])
        out[sensor_cols] = scaler.transform(out[sensor_cols])

    return out, le, scaler, sensor_cols


def compute_features(window, sensor_cols, fs):
    """Extract 13 features (time + freq domain) from one window per sensor channel."""
    feats = {}
    for col in sensor_cols:
        x = window[col].values

        # time domain
        feats[f'{col}_mean'] = np.mean(x)
        feats[f'{col}_std']  = np.std(x)
        feats[f'{col}_min']  = np.min(x)
        feats[f'{col}_max']  = np.max(x)
        feats[f'{col}_range'] = np.ptp(x)
        feats[f'{col}_skew'] = stats.skew(x)
        feats[f'{col}_kurt'] = stats.kurtosis(x)
        feats[f'{col}_rms']  = np.sqrt(np.mean(x ** 2))
        feats[f'{col}_zcr']  = ((x[:-1] * x[1:]) < 0).sum() / len(x)

        # freq domain — FFT
        fft_vals = fft(x)
        fft_freq = fftfreq(len(x), 1 / fs)
        power = np.abs(fft_vals) ** 2
        pos_pwr = power[1:len(x) // 2]
        pos_frq = fft_freq[1:len(x) // 2]

        if len(pos_pwr) > 0 and pos_pwr.max() > 0:
            feats[f'{col}_fft_dom_freq'] = pos_frq[np.argmax(pos_pwr)]
        else:
            feats[f'{col}_fft_dom_freq'] = 0.0

        # Welch PSD
        _, psd = welch(x, fs=fs, nperseg=len(x), scaling='density')
        feats[f'{col}_psd_mean'] = np.mean(psd)
        feats[f'{col}_psd_peak'] = np.max(psd)

        # signal energy
        feats[f'{col}_signal_energy'] = np.sum(x ** 2) / len(x)

    return feats


def build_windows(df, sensor_cols, win=WIN_LEN, step=STEP, fs=FS):
    """Slide a window across each subject's data, extract features + majority label."""
    all_X, all_y = [], []

    for subj in df['subject'].unique():
        sdf = df[df['subject'] == subj].reset_index(drop=True)
        if len(sdf) < win:
            continue
        for start in range(0, len(sdf) - win + 1, step):
            w = sdf.iloc[start:start + win]
            lbl = stats.mode(w['label'])[0]
            all_X.append(compute_features(w, sensor_cols, fs))
            all_y.append(lbl)

    return pd.DataFrame(all_X), np.array(all_y, dtype=int)


def run_split(subjects, name, le=None, scaler=None):
    """Full sub-pipeline: load → clean → scale → window → features."""
    print(f"\n--- {name} pipeline (subjects {subjects}) ---")
    fit = (name == 'Train')

    raw = load_data(DATA_DIR, subjects)
    cleaned = clean(raw)
    print(f"[PREPROC] {'Fitting' if fit else 'Transforming'} scaler/encoder ...")
    processed, le_out, sc_out, sensor_cols = scale_and_encode(cleaned, le, scaler, fit=fit)

    X, y = build_windows(processed, sensor_cols)
    print(f"[FEAT] Done — {len(y):,} windows, X shape {X.shape}")
    return X, y, sc_out, le_out


def main():
    for d in [OUTPUT_DIR, CKPT_DIR, os.path.join(OUTPUT_DIR, 'logs')]:
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, 'logs', 'preprocess_output.txt')

    with open(log_path, 'w') as f:
        with redirect_stdout(f):
            train_subj = list(range(1, 8))   # subjects 1-7
            test_subj  = list(range(8, 11))  # subjects 8-10

            print("=========================================================")
            print("       MHEALTH DATA PREPROCESSING")
            print(f"Train subjects: {train_subj}")
            print(f"Test subjects:  {test_subj}")
            print(f"Window: {WIN_LEN} samples @ {FS} Hz | Step: {STEP} (50% overlap)")

            X_tr, y_tr, scaler, le = run_split(train_subj, 'Train')

            print("\n[INFO] Label mapping:")
            for orig, idx in zip(le.classes_, range(len(le.classes_))):
                print(f"  {ACTIVITY_MAP.get(orig, '?')} ({orig}) -> {idx}")

            X_te, y_te, _, _ = run_split(test_subj, 'Test', le=le, scaler=scaler)

            # save everything
            X_tr.to_csv(os.path.join(OUTPUT_DIR, 'X_train_features.csv'), index=False)
            X_te.to_csv(os.path.join(OUTPUT_DIR, 'X_test_features.csv'), index=False)
            np.savetxt(os.path.join(OUTPUT_DIR, 'y_train_labels.csv'), y_tr, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(OUTPUT_DIR, 'y_test_labels.csv'), y_te, delimiter=',', fmt='%d')

            with open(os.path.join(CKPT_DIR, 'scaler_train.pkl'), 'wb') as fp:
                pickle.dump(scaler, fp)
            with open(os.path.join(CKPT_DIR, 'label_encoder_train.pkl'), 'wb') as fp:
                pickle.dump(le, fp)

            print(f"\n[SAVE] Train: X {X_tr.shape}, y {y_tr.shape}")
            print(f"[SAVE] Test:  X {X_te.shape}, y {y_te.shape}")
            print("[SAVE] Saved scaler + label encoder to checkpoints/")
            print("=========================================================")

    print(f"Preprocessing done. Log saved to {log_path}")


if __name__ == "__main__":
    main()
