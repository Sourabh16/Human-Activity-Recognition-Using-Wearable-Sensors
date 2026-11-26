# Human Activity Recognition Using Wearable Sensors

SVM-based Human Activity Recognition on the [MHEALTH dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset). Classifies 12 physical activities from chest, ankle, and arm sensor data using a Support Vector Classifier with RBF kernel.

**Test accuracy: 94.92%** on completely unseen subjects (subject-independent split).

> Course project for CSL7620 — Machine Learning, IIT Jodhpur  
> Authors: Sourabh Singh, Ranvir Singh

---

## Project Structure

```
├── src/
│   ├── preprocessing.py    # Data loading, cleaning, feature extraction
│   ├── train.py             # SMOTE + GridSearchCV + SVM training
│   ├── evaluate.py          # Confusion matrix & PCA correlation plots
│   └── ablation.py          # Sensor-location ablation study
├── report/
│   ├── main.tex             # CVPR-format LaTeX source (8 pages)
│   ├── main.pdf             # Compiled report
│   ├── cvpr.sty             # CVPR style file
│   └── figures/             # Figures embedded in the report
├── checkpoints/             # Trained model & fitted scaler/encoder (generated)
├── outputs/                 # Feature CSVs, predictions, plots, logs (generated)
├── requirements.txt         # Python dependencies
├── REPRODUCIBILITY.md       # Reproducibility declaration & instructions
└── README.md                # This file
```

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Sourabh16/Human-Activity-Recognition-Using-Wearable-Sensors.git
cd Human-Activity-Recognition-Using-Wearable-Sensors
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Download the dataset

Download the MHEALTH dataset from the UCI repository:  
👉 https://archive.ics.uci.edu/dataset/319/mhealth+dataset

Extract the ZIP and place the `MHEALTHDATASET` folder inside a `dataset/` directory at the project root:

```
dataset/
└── MHEALTHDATASET/
    ├── mHealth_subject1.log
    ├── mHealth_subject2.log
    ├── ...
    └── mHealth_subject10.log
```

Alternatively, set the environment variable:
```bash
export MHEALTH_DATA_DIR=/path/to/your/MHEALTHDATASET
```

### 4. Run the pipeline

Scripts must be run **in order** from the `src/` directory:

```bash
cd src

# Step 1: Preprocessing (~1–2 min)
python preprocessing.py

# Step 2: Training (~5–15 min depending on hardware)
python train.py

# Step 3: Evaluation & plots (~10 seconds)
python evaluate.py

# Step 4 (Optional): Ablation study (~20–45 min)
python ablation.py
```

### 5. Check results

After running the pipeline, outputs are saved in:

| Path | Contents |
|------|----------|
| `outputs/logs/` | Console logs from each step |
| `outputs/metrics/` | `classification_report.json` |
| `outputs/figures/` | `confusion_matrix.png`, `reduced_ankle_vs_others_correlation.png` |
| `outputs/` | Feature CSVs and prediction CSVs |
| `checkpoints/` | `svm_model.pkl`, `scaler_train.pkl`, `label_encoder_train.pkl` |

---

## Results

| Metric | Value |
|--------|-------|
| Best hyperparameters | C=10, kernel=RBF, gamma=scale |
| Cross-validation accuracy | 95.13% (±0.66%) |
| **Test accuracy (unseen subjects)** | **94.92%** |
| Number of features | 273 (21 channels × 13 features) |
| Training windows | 1,879 (Subjects 1–7) |
| Test windows | 788 (Subjects 8–10) |

### Ablation Study

| Scenario | Accuracy | Change |
|----------|----------|--------|
| Baseline (all sensors) | 91.24% | — |
| Remove chest | 88.83% | −2.64% |
| Remove arm | 89.59% | −1.81% |
| Remove ankle | **92.26%** | **+1.11%** |

Key finding: Removing ankle sensor features *improves* accuracy, indicating sensor redundancy.

---

## Dataset

- **Name:** MHEALTH (Mobile Health)
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- **Subjects:** 10 volunteers
- **Activities:** 12 classes
- **Sensors:** Chest (accelerometer), Left ankle (acc + gyro + mag), Right arm (acc + gyro + mag), Chest ECG
- **Sampling rate:** 50 Hz

The dataset is **not included** in this repository due to its size (~220 MB). Download it from the link above.

---

## Checkpoints

Model checkpoints (`svm_model.pkl`, ~1.8 MB) are small enough to regenerate by running the pipeline. They are not included in the repository to ensure reproducibility from scratch. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for details.

---

## Report

The 8-page CVPR-format report is available at [`report/main.pdf`](report/main.pdf). To recompile:

```bash
cd report
tectonic main.tex
```

---

## License

This project is for academic purposes (CSL7620, IIT Jodhpur). The MHEALTH dataset is provided by its original authors under the UCI ML Repository terms.
