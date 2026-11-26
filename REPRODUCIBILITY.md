# Reproducibility Declaration & Instructions

## Declaration

We declare that all results reported in this project are fully reproducible using the code, data, and instructions provided in this repository. The pipeline is deterministic given the same dataset, random seeds, and software versions.

**Random seed:** `42` (used in SMOTE, SVM, and ablation scripts)  
**Data split:** Subject-dependent — Subjects 1–7 for training, 8–10 for testing (hardcoded, no randomness)  
**Date of reported results:** November 27, 2026

---

## Reproduction Steps

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~500 MB free disk space

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/Sourabh16/Human-Activity-Recognition-Using-Wearable-Sensors.git
cd Human-Activity-Recognition-Using-Wearable-Sensors

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. Install dependencies (exact versions pinned)
pip install -r requirements.txt

# 4. Download MHEALTH dataset
# Visit: https://archive.ics.uci.edu/dataset/319/mhealth+dataset
# Extract mhealth+dataset.zip into dataset/MHEALTHDATASET/
# You should have: dataset/MHEALTHDATASET/mHealth_subject{1..10}.log

# 5. Run the full pipeline
cd src
python preprocessing.py     # ~1–2 min
python train.py              # ~5–15 min
python evaluate.py           # ~10 sec

# 6. (Optional) Run ablation study
python ablation.py           # ~20–45 min
```

### Expected Results

After running the pipeline with the MHEALTH dataset:

| Metric | Expected Value |
|--------|---------------|
| Training windows (X_train) | 1,879 × 273 |
| Test windows (X_test) | 788 × 273 |
| SMOTE resampled count | 2,052 |
| Best hyperparameters | C=10, kernel=rbf, gamma=scale |
| CV accuracy | ~95.13% (±0.66%) |
| **Test accuracy** | **~94.92%** |

Minor variations (<0.5%) may occur due to differences in floating-point arithmetic across platforms, but the overall results should be consistent.

---

## Software Environment

The results were produced on the following environment:

| Component | Version |
|-----------|---------|
| OS | macOS (Apple Silicon) |
| Python | 3.12.7 |
| scikit-learn | 1.5.2 |
| imbalanced-learn | 0.12.4 |
| numpy | 2.0.2 |
| pandas | 2.2.2 |
| scipy | 1.13.1 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |

All versions are pinned in `requirements.txt`.

---

## Checkpoints

Model checkpoints (`svm_model.pkl`, `scaler_train.pkl`, `label_encoder_train.pkl`) are **not** included in the repository. They are small (<2 MB total) and are regenerated deterministically by running `preprocessing.py` followed by `train.py`.

**Justification for exclusion:** The checkpoints are fully reproducible from the source code and dataset. Including them would add unnecessary binary files to version control. The total regeneration time is under 15 minutes on commodity hardware.

---

## Dataset

The MHEALTH dataset is **not** included in this repository due to its size (~220 MB uncompressed).

**Download:** https://archive.ics.uci.edu/dataset/319/mhealth+dataset  
**Citation:** Banos et al., "mHealthDroid: a novel framework for agile development of mobile health applications," Int. Workshop on Ambient Assisted Living, Springer, 2014.

Place the extracted files in `dataset/MHEALTHDATASET/` or set the `MHEALTH_DATA_DIR` environment variable.

---

## Directory Structure After Full Run

```
├── src/                         # Source code (versioned)
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── ablation.py
├── dataset/                     # NOT in git (download separately)
│   └── MHEALTHDATASET/
│       ├── mHealth_subject1.log
│       └── ...
├── outputs/                     # Generated (NOT in git)
│   ├── X_train_features.csv
│   ├── X_test_features.csv
│   ├── y_train_labels.csv
│   ├── y_test_labels.csv
│   ├── y_true.csv
│   ├── y_pred.csv
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── reduced_ankle_vs_others_correlation.png
│   ├── logs/
│   │   ├── preprocess_output.txt
│   │   ├── train_output.txt
│   │   └── evaluation_output.txt
│   └── metrics/
│       └── classification_report.json
├── checkpoints/                 # Generated (NOT in git)
│   ├── svm_model.pkl
│   ├── scaler_train.pkl
│   └── label_encoder_train.pkl
├── report/
│   ├── main.tex
│   ├── main.pdf
│   └── cvpr.sty
├── requirements.txt
├── REPRODUCIBILITY.md
├── README.md
└── .gitignore
```
