"""
Brain StrokeSense — Data Preprocessing Module
=============================================
Loads and validates the stroke prediction dataset, builds a leakage-safe
preprocessing pipeline, and produces stratified train/validation/test splits.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (relative to this file so the app works from any CWD)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
DATASET_PATH = DATA_DIR / "healthcare-dataset-stroke-data.csv"

REQUIRED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

TARGET_COLUMN = "stroke"
ID_COLUMN = "id"
RANDOM_STATE = 42

NUMERICAL_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def ensure_directories() -> None:
    """Create project output directories if they do not exist."""
    for path in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, GRAPHS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the stroke prediction CSV and validate its schema.

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: if required columns are absent.
    """
    csv_path = Path(path) if path else DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Download the Stroke Prediction Dataset from:\n"
            "  https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset\n"
            "Save the file as 'healthcare-dataset-stroke-data.csv' inside the 'data/' folder."
        )

    df = pd.read_csv(csv_path)

    # BMI is often stored as the string "N/A" in this dataset
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset schema mismatch. Missing required columns: {missing_cols}.\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Basic cleaning before train/test split:
    - Drop exact duplicate rows
    - Normalize categorical string casing/whitespace
    - Replace known invalid BMI markers (already coerced to NaN in load)
    - Flag unusual categories (kept; OneHotEncoder will handle them)

    Outliers are NOT removed automatically. Extreme health measurements
    (very high glucose / BMI, very young or old age) can be clinically
    legitimate and are retained for transparency.
    """
    summary: Dict[str, Any] = {}
    working = df.copy()

    summary["rows_before"] = int(len(working))
    summary["duplicate_rows"] = int(working.duplicated().sum())
    working = working.drop_duplicates().reset_index(drop=True)
    summary["rows_after_dedup"] = int(len(working))

    # Normalize categorical text
    for col in CATEGORICAL_FEATURES:
        if col in working.columns and working[col].dtype == object:
            working[col] = working[col].astype(str).str.strip()

    # Gender: uncommon values such as "Other" are kept (rare but valid)
    summary["gender_unique"] = (
        working["gender"].value_counts(dropna=False).to_dict()
        if "gender" in working.columns
        else {}
    )

    # Missing-value report (before imputation)
    missing = working.isna().sum()
    summary["missing_before_imputation"] = {
        col: int(missing[col]) for col in working.columns if missing[col] > 0
    }
    summary["bmi_missing_count"] = int(working["bmi"].isna().sum()) if "bmi" in working.columns else 0

    # Outlier inspection (IQR flags only — values are kept)
    outlier_flags: Dict[str, int] = {}
    for col in NUMERICAL_FEATURES:
        series = working[col].dropna()
        if series.empty:
            outlier_flags[col] = 0
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_flags[col] = int(((series < lower) | (series > upper)).sum())
    summary["iqr_outlier_flags_retained"] = outlier_flags
    summary["outlier_policy"] = (
        "IQR outliers are inspected and reported but NOT removed, because "
        "extreme BMI, glucose, or age values may be clinically legitimate."
    )

    return working, summary


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes numerical features (median) and scales them
    - Imputes binary features (most frequent)
    - Imputes + one-hot encodes categorical features

    Fitted ONLY on training data later to prevent leakage.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return human-readable feature names after encoding."""
    names: List[str] = []
    names.extend(NUMERICAL_FEATURES)
    names.extend(BINARY_FEATURES)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        names.extend(cat_names)
    except Exception:
        names.extend(CATEGORICAL_FEATURES)
    return names


def prepare_splits(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Full preprocessing workflow:
    1. Load & validate
    2. Clean
    3. Separate X / y (drop id)
    4. Stratified train / val / test split
    5. Fit preprocessor on TRAIN only
    6. Transform all splits
    7. Persist pipeline + summary JSON

    Returns a dictionary with raw and transformed splits plus metadata.
    """
    ensure_directories()

    if df is None:
        df = load_dataset()

    cleaned, clean_summary = clean_dataframe(df)

    X = cleaned[FEATURE_COLUMNS].copy()
    y = cleaned[TARGET_COLUMN].astype(int).copy()

    # First split: hold out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Second split: train vs validation from remaining data
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val,
        stratify=y_temp,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    # Persist fitted preprocessor
    pipeline_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_path)

    # Class distribution summary
    def _dist(series: pd.Series) -> Dict[str, Any]:
        counts = series.value_counts().to_dict()
        total = int(len(series))
        return {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {
                str(k): round(100.0 * v / total, 2) for k, v in counts.items()
            },
            "total": total,
        }

    summary: Dict[str, Any] = {
        "dataset_path": str(DATASET_PATH),
        "n_rows": int(len(cleaned)),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "encoded_feature_names": feature_names,
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "class_distribution": {
            "overall": _dist(y),
            "train": _dist(y_train),
            "validation": _dist(y_val),
            "test": _dist(y_test),
        },
        "cleaning": clean_summary,
        "imbalance_strategy": (
            "Primary strategy: class_weight='balanced' in supported classifiers. "
            "Optional SMOTE is applied only on training data inside training "
            "experiments and compared against the class-weight baseline."
        ),
        "leakage_prevention": (
            "Preprocessor (imputer, scaler, encoder) is fit exclusively on the "
            "training split, then applied to validation/test. SMOTE, if used, "
            "is applied only after the train split and only to training folds."
        ),
        "random_state": random_state,
        "preprocessor_path": str(pipeline_path),
    }

    summary_path = OUTPUTS_DIR / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_t": X_train_t,
        "X_val_t": X_val_t,
        "X_test_t": X_test_t,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "summary": summary,
        "cleaned_df": cleaned,
    }


def get_dataset_overview(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Compute overview statistics for the Dataset Explorer page."""
    ensure_directories()
    if df is None:
        df = load_dataset()

    cleaned, clean_summary = clean_dataframe(df)
    overview: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "describe_numeric": json.loads(
            df.select_dtypes(include=[np.number])
            .describe()
            .round(3)
            .to_json()
        ),
        "missing": {col: int(v) for col, v in df.isna().sum().items()},
        "missing_pct": {
            col: round(100.0 * float(v) / len(df), 2)
            for col, v in df.isna().sum().items()
        },
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": {
            str(k): int(v)
            for k, v in df[TARGET_COLUMN].value_counts().items()
        },
        "target_percentages": {
            str(k): round(100.0 * v / len(df), 2)
            for k, v in df[TARGET_COLUMN].value_counts().items()
        },
        "cleaning": clean_summary,
        "n_after_clean": int(len(cleaned)),
    }

    out_path = OUTPUTS_DIR / "dataset_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, default=str)

    return overview


def load_preprocessor() -> ColumnTransformer:
    """Load the persisted preprocessing pipeline."""
    path = MODELS_DIR / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. Run train_model.py first."
        )
    return joblib.load(path)


if __name__ == "__main__":
    print("=" * 60)
    print("Brain StrokeSense — Preprocessing")
    print("=" * 60)
    overview = get_dataset_overview()
    print(f"Dataset shape: {overview['shape']}")
    print(f"Missing values: {overview['missing']}")
    print(f"Duplicates: {overview['duplicates']}")
    print(f"Target distribution: {overview['target_distribution']}")

    splits = prepare_splits()
    print(f"Train / Val / Test sizes: {splits['summary']['split_sizes']}")
    print(f"Encoded feature count: {len(splits['feature_names'])}")
    print(f"Summary saved to: {OUTPUTS_DIR / 'preprocessing_summary.json'}")
    print("Preprocessing complete.")
