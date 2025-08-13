# Brain Resilience Prediction

A pre-trained machine learning model for predicting **brain resilience** (High vs. Low) using **structural MRI** features from FreeSurfer and FastSurfer.  
This model achieved the highest AUC in the thesis experiments and comes with a complete inference script (`predict.py`). Users only need to provide two raw feature files to get predictions.

---

## 📦 Contents
- `md.joblib` — The trained best model (**includes feature list and fitted StandardScaler**)
- `predict.py` — Inference script (automatically merges two input files, selects features, standardizes, and predicts)
- `Testing_freesurfer.csv` — Example FreeSurfer features file
- `Testing_fastsurfer.csv` — Example FastSurfer features file
- `requirements.txt` — Required Python packages and versions

---

## ⚙️ Installation
1. Install **Python 3.9–3.11** (recommended: 3.11)
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If pip is not recognized, use:
   ```powershell
   python -m pip install -r requirements.txt
   ```
---

## 📂 Input File Format

### 1. FreeSurfer features file 
- Contains all FreeSurfer numeric features
- Must have an `ID` column (subject ID)
- The script automatically removes the column `Sum` if present

### 2. FastSurfer features file 
- Contains all FastSurfer numeric features
- Must have the same `ID` column as the FreeSurfer file
- The script automatically removes columns `Sum` and `label_y` if present

The two files are **inner merged** on the `ID` column.

---

## 🚀 How to Run 

```
python .\predict.py `
  --model ".\md.joblib" `
  --freesurfer ".\Testing_freesurfer.csv" `
  --fastsurfer ".\Testing_fastsurfer.csv" `
  --output ".\result.csv" `
  --proba `
  --id-col ID
```

## 🚀 If the multi-line format causes issues, please try this single-line format

```
python predict.py --model ".\md.joblib" --freesurfer ".\Testing_freesurfer.csv" --fastsurfer ".\Testing_fastsurfer.csv" --output ".\result.csv" --proba --id-col ID
```

---

## 🔧 Parameters
| Parameter | Description |
|-----------|-------------|
| `--model` | Path to the `.joblib` model file (includes classifier, feature list, and scaler) |
| `--freesurfer` | Path to FreeSurfer features CSV file |
| `--fastsurfer` | Path to FastSurfer features CSV file |
| `--output` | Output CSV file name (default: `predictions.csv`) |
| `--proba` | Include the positive class probability (`prob_1`) in the output |
| `--id-col` | Name of the ID column (default: `ID`) |
| `--encoding` | CSV file encoding (default: `utf-8-sig`) |

---

## 📄 Output Format
The output CSV contains:
- `ID` — Original subject ID
- `pred` — Predicted class (`1` = High resilience, `0` = Low resilience)
- `prob_1` — Positive class probability (only if `--proba` is used)

Example:
| ID   | pred | prob_1 |
|------|------|--------|
| S001 | 1    | 0.873  |
| S002 | 0    | 0.421  |

---

## 📌 Notes
- Ensure both input files have the same `ID` column values and format.
- The provided `md.joblib` already contains the feature list and fitted scaler used during training.
- Do **not** re-standardize the data before prediction — the script handles this using the original training scaler.
