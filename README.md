# Brain StrokeSense

**AI-Based Brain Stroke Risk Prediction and Data Analytics System**

Educational college project (B.Tech AI & Data Science) that combines data preprocessing, exploratory data analysis for every dataset attribute, multi-model machine learning, evaluation, explainability, and a modern Flask website.

> **Medical safety:** This is an educational prototype, **not** a clinically validated medical device. Predictions are **not** diagnoses. Model scores are **not** calibrated probabilities of future stroke. Do not use for treatment or emergency decisions. Stroke symptoms require immediate emergency care.

---

## Problem Statement

Early identification of stroke-related risk patterns is challenging because multiple demographic, medical, and lifestyle factors may influence recorded stroke outcomes.

## Proposed Solution

An AI-based web application that analyses patient health data, performs preprocessing and EDA, compares multiple ML algorithms, and demonstrates predictions through an interactive local website.

 

## Features

- Dataset explorer with schema, missing values, pagination  
- Leakage-safe preprocessing (train-only fit)  
- Graphs for **every** dataset attribute  
- Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting  
- Optional SMOTE experiment (train-only) vs class-weight baseline  
- Validation-F1 model selection; single untouched test evaluation  
- Permutation feature importance  
- Interactive prediction demo with safety disclaimers  
- Research gaps & conclusion page  

## Technology Stack

Python · Flask · HTML5 · CSS3 · JavaScript · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn · Joblib · Chart.js (CDN)

## Folder Structure

```
BrainStrokeSense/
  app.py
  requirements.txt
  README.md
  train_model.py
  preprocess.py
  analyze_data.py
  evaluate_models.py
  data/
    healthcare-dataset-stroke-data.csv
  models/
    trained_model.joblib
    preprocessor.joblib
    model_metadata.json
    model_comparison.csv
    ...
  outputs/
    dataset_summary.json
    preprocessing_summary.json
    model_metrics.json
    eda_catalog.json
    feature_importance.json
    graphs/
      *.png
  templates/
    base.html, index.html, dataset.html, preprocessing.html,
    analysis.html, models.html, predict.html, explainability.html, research.html
  static/
    css/style.css
    js/main.js
```

## Dataset Source & Setup

**Source:** [Stroke Prediction Dataset (Kaggle — fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**Expected file:** `data/healthcare-dataset-stroke-data.csv`

### How to download (Windows)

1. Open the Kaggle link above and download the CSV (Kaggle account required).  
2. Or use any trusted mirror of the same public dataset.  
3. Save / rename the file to:  
   `BrainStrokeSense/data/healthcare-dataset-stroke-data.csv`

Required columns:  
`id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke`

The `id` column is excluded from predictive features. Target = `stroke`.

## Preprocessing Methodology

- Validate schema; coerce BMI (`N/A` → missing)  
- Drop exact duplicates  
- **Do not auto-remove IQR outliers** (extreme health values may be legitimate)  
- Stratified train / validation / test split  
- Median / mode imputation, OneHotEncoder, StandardScaler — **fit on train only**  
- Class imbalance: `class_weight='balanced'`; optional SMOTE on training data only  

## EDA Methodology

Automated charts for numerical (hist/box/KDE/stroke compare), categorical (counts/percent/crosstab), target imbalance, missingness, dtypes, correlation heatmap, and before/after dedup. Catalog saved to `outputs/eda_catalog.json`.

## Model Training & Evaluation

- Classifiers: LR, Decision Tree, RF, SVM, Gradient Boosting (+ optional LR+SMOTE)  
- Selection metric: **validation F1** (positive class)  
- Final report: one evaluation on untouched **test** set  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, PR curves  
- Explainability: permutation importance on validation F1  

**Accuracy warning:** With rare stroke labels, high accuracy can hide poor minority-class detection.

## Run on Windows (VS Code)

### 1. Open the project

```powershell
cd C:\Users\mudal\BrainStrokeSense
```

Open this folder in Visual Studio Code.

### 2. Create & activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Generate EDA graphs

```powershell
python analyze_data.py
```

### 5. Train models, evaluate, and build explainability artifacts

```powershell
python evaluate_models.py
```

(`evaluate_models.py` also runs training. You may run `python train_model.py` alone if needed.)

### 6. Start the website

```powershell
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Website Pages

| Page | Route | Purpose |
|------|-------|---------|
| Home | `/` | Brand, problem, objectives, live stats |
| Dataset | `/dataset` | Schema, quality, paginated preview |
| Preprocessing | `/preprocessing` | Pipeline explanation & summaries |
| Analysis | `/analysis` | Gallery of all EDA charts |
| Models | `/models` | Comparison table & evaluation plots |
| Predict | `/predict` | Interactive demo form |
| Explain | `/explainability` | Feature importance |
| Research | `/research` | Gaps, limitations, conclusion |

## Presentation Workflow (short)

1. Introduce problem & objectives (Home)  
2. Show dataset quality (Dataset)  
3. Explain leakage-safe preprocessing  
4. Walk through attribute graphs (Analysis)  
5. Compare models; justify validation-F1 selection  
6. Demo prediction with safety disclaimer  
7. Discuss importance ≠ causation  
8. Close with limitations & future scope  

## Limitations

- Not clinically validated  
- Historical labels ≠ future individual risk guarantees  
- Class imbalance, missing BMI, unknown smoking categories  
- No probability calibration study  
- No regulatory / fairness / multi-site validation  

## Future Enhancements

Calibration, external validation, fairness audits, richer features under ethics approval, optional SHAP, privacy-preserving deployment research.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Dataset not found` | Place CSV in `data/healthcare-dataset-stroke-data.csv` |
| `Trained model not found` | Run `python evaluate_models.py` |
| Blank Analysis page | Run `python analyze_data.py` |
| `ModuleNotFoundError` | Activate `.venv` and `pip install -r requirements.txt` |
| Port 5000 in use | Change port in `app.py` or stop the other process |
| Charts missing after restart | Charts live in `outputs/graphs/` — regenerate if deleted |

## License / Academic Use

Built for academic demonstration. Dataset terms remain those of the original Kaggle publisher. Always cite the dataset source in your report.
