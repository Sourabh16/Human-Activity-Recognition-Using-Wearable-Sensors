"""
ablation.py
Sensor-location ablation study: systematically removes one sensor group
at a time (chest / arm / ankle) and re-trains the SVM to measure each
sensor's contribution to classification accuracy.

Authors: Sourabh Singh, Ranvir Singh
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    'MHEALTH_DATA_DIR',
    os.path.join(BASE_DIR, '..', 'dataset', 'MHEALTHDATASET')
)

TRAIN_SUBJ = list(range(1, 8))
TEST_SUBJ  = list(range(8, 11))

SENSOR_COLS = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
]
ALL_COLS = SENSOR_COLS + ['ecg_lead1', 'ecg_lead2', 'label']

ACTIVITY_MAP = {
    1: "Standing still", 2: "Sitting and relaxing", 3: "Lying down",
    4: "Walking", 5: "Climbing stairs", 6: "Waist bends forward",
    7: "Frontal elevation of arms", 8: "Knees bending (crouching)",
    9: "Cycling", 10: "Jogging", 11: "Running", 12: "Jump front & back"
}


# --- pipeline helpers (self-contained for ablation) ---

def load_data(subjects):
    frames = []
    for s in subjects:
        fpath = os.path.join(DATA_DIR, f'mHealth_subject{s}.log')
        if not os.path.exists(fpath):
            print(f"  [WARN] {fpath} not found — generating mock data for subject {s}")
            n = 20000
            df = pd.DataFrame({c: np.random.randn(n) for c in ALL_COLS if c != 'label'})
            df['label'] = np.random.choice(list(ACTIVITY_MAP.keys()) + [0], n)
            mask = np.random.choice([False, True], size=df[SENSOR_COLS].shape, p=[0.999, 0.001])
            df[SENSOR_COLS] = df[SENSOR_COLS].mask(mask, other=np.nan)
            df['subject'] = s
            frames.append(df)
            continue
        df = pd.read_csv(fpath, sep=r'\s+', header=None, names=ALL_COLS)
        df['subject'] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def preprocess_fit(df):
    """Clean + fit scaler/encoder on training data."""
    df = df[df['label'] != 0].copy()
    df.drop(columns=['ecg_lead1', 'ecg_lead2'], errors='ignore', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    scols = [c for c in df.columns if c not in ('label', 'subject')]
    scaler = StandardScaler()
    df[scols] = scaler.fit_transform(df[scols])
    return df, le, scaler


def preprocess_transform(df, le, scaler):
    """Clean + transform test data with fitted objects."""
    df = df[df['label'] != 0].copy()
    df.drop(columns=['ecg_lead1', 'ecg_lead2'], errors='ignore', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    def safe_enc(v):
        try: return le.transform([v])[0]
        except ValueError: return -1
    df['label'] = df['label'].apply(safe_enc)
    df = df[df['label'] != -1].copy()

    scols = [c for c in df.columns if c not in ('label', 'subject')]
    if scols and not df.empty:
        df[scols] = scaler.transform(df[scols])
    return df


def extract_features(window, sensor_cols, fs=50):
    feats = {}
    for col in sensor_cols:
        x = window[col].values
        feats[f'{col}_mean'] = np.mean(x)
        feats[f'{col}_std']  = np.std(x)
        feats[f'{col}_min']  = np.min(x)
        feats[f'{col}_max']  = np.max(x)
        feats[f'{col}_rms']  = np.sqrt(np.mean(x**2))
        if len(x) > 1:
            feats[f'{col}_skew'] = stats.skew(x)
            feats[f'{col}_kurt'] = stats.kurtosis(x)
            feats[f'{col}_zcr']  = ((x[:-1] * x[1:]) < 0).sum() / len(x)
        else:
            feats[f'{col}_skew'] = feats[f'{col}_kurt'] = feats[f'{col}_zcr'] = 0

        feats[f'{col}_signal_energy'] = np.sum(x**2) / len(x)
        try:
            fv = fft(x); ff = fftfreq(len(x), 1/fs)
            pw = np.abs(fv)**2
            feats[f'{col}_fft_dom_freq'] = ff[np.argmax(pw[1:]) + 1]
            _, psd = welch(x, fs=fs, nperseg=len(x), scaling='density')
            feats[f'{col}_psd_mean'] = np.mean(psd)
            feats[f'{col}_psd_peak'] = np.max(psd)
        except Exception:
            feats[f'{col}_fft_dom_freq'] = 0
            feats[f'{col}_psd_mean'] = feats[f'{col}_psd_peak'] = 0
    return feats


def build_windows(df, sensor_cols, win=256, step=128, fs=50):
    X, y = [], []
    for s in df['subject'].unique():
        sdf = df[df['subject'] == s].reset_index(drop=True)
        if len(sdf) < win:
            continue
        for i in range(0, len(sdf) - win + 1, step):
            w = sdf.iloc[i:i + win]
            lbl = stats.mode(w['label'])[0]
            X.append(extract_features(w, sensor_cols, fs))
            y.append(lbl)
    return pd.DataFrame(X), np.array(y, dtype=int)


def train_and_eval(X_tr, y_tr, X_te, y_te):
    """SMOTE + GridSearchCV + evaluate — compact version for ablation."""
    counts = pd.Series(y_tr).value_counts()
    k = min(5, max(1, counts.min() - 1))

    Xr, yr = X_tr, y_tr
    if counts.min() > 1:
        try:
            sm = SMOTE(k_neighbors=k, random_state=42)
            Xr, yr = sm.fit_resample(X_tr, y_tr)
        except ValueError as e:
            print(f"   [WARN] SMOTE failed: {e}")

    grid = {'C': [1, 10], 'kernel': ['rbf', 'poly'], 'gamma': ['scale', 'auto']}
    gs = GridSearchCV(SVC(random_state=42), grid, cv=3,
                      scoring='accuracy', n_jobs=-1)
    gs.fit(Xr, yr)
    best = gs.best_estimator_

    acc = accuracy_score(y_te, best.predict(X_te))
    print(f"   Best: {gs.best_params_}  |  Train acc: {best.score(Xr, yr):.4f}")
    return acc


def run_experiment(sensor_list, full_df):
    """Run one ablation scenario end-to-end."""
    tr_df = full_df[full_df['subject'].isin(TRAIN_SUBJ)].copy()
    te_df = full_df[full_df['subject'].isin(TEST_SUBJ)].copy()

    tr_proc, le, scaler = preprocess_fit(tr_df)
    te_proc = preprocess_transform(te_df, le, scaler)

    tr_filt = tr_proc[['subject', 'label'] + sensor_list]
    te_filt = te_proc[['subject', 'label'] + sensor_list]

    X_tr, y_tr = build_windows(tr_filt, sensor_list)
    X_te, y_te = build_windows(te_filt, sensor_list)

    if X_tr.empty or X_te.empty:
        print("   [ERROR] Not enough data. Skipping.")
        return 0.0, 0

    return train_and_eval(X_tr.values, y_tr, X_te.values, y_te), X_tr.shape[1]


def main():
    scenarios = {
        "Baseline (All Sensors)":
            SENSOR_COLS,
        "Ablate Chest Sensor (3 cols)":
            [c for c in SENSOR_COLS if not c.startswith('chest_')],
        "Ablate Arm Sensor (9 cols)":
            [c for c in SENSOR_COLS if not c.startswith('arm_')],
        "Ablate Ankle Sensor (9 cols)":
            [c for c in SENSOR_COLS if not c.startswith('ankle_')],
    }

    print("Loading all subjects ...")
    full_df = load_data(TRAIN_SUBJ + TEST_SUBJ)
    if full_df.empty:
        print("[FATAL] No data loaded.")
        return

    results = {}
    print("\n--- Ablation Study (Sensor Location) ---")
    for name, cols in scenarios.items():
        print(f"\n[EXP] {name}  ({len(cols)} raw cols)")
        acc, n_feat = run_experiment(cols, full_df)
        results[name] = {'accuracy': acc, 'n_features': n_feat}
        print(f"  => Test accuracy: {acc:.4f}")

    # print summary table
    baseline = results['Baseline (All Sensors)']['accuracy']
    print("\n=========================================================")
    print("          ABLATION RESULTS")
    print("=========================================================")
    print(f"{'Scenario':<35} {'Acc':>8} {'Feats':>7} {'Change':>9}")
    print("-" * 62)
    for name, r in results.items():
        drop = 100 * (baseline - r['accuracy']) / baseline if baseline > 0 else 0
        tag = "Baseline" if name.startswith("Baseline") else f"{drop:+.2f}%"
        print(f"{name:<35} {r['accuracy']:>8.4f} {r['n_features']:>7} {tag:>9}")
    print("=========================================================\n")


if __name__ == "__main__":
    np.random.seed(42)
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        print("Make sure all dependencies are installed (see requirements.txt).")
