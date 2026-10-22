"""
Brain StrokeSense — Model Training
==================================
Trains Logistic Regression, Decision Tree, Random Forest, SVM, and optional
Gradient Boosting with class weights. Selects the best model using validation
F1-score (stroke-positive class), then evaluates once on the untouched test set.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from preprocess import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    ensure_directories,
    prepare_splits,
)

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


def _metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None
        try:
            out["average_precision"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            out["average_precision"] = None
    else:
        out["roc_auc"] = None
        out["average_precision"] = None
    return out


def _get_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Min-max to [0, 1] for ranking metrics only (not calibrated probability)
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.zeros_like(scores, dtype=float)
        return (scores - mn) / (mx - mn)
    return np.zeros(len(X), dtype=float)


def build_candidate_models() -> Dict[str, Any]:
    """Return estimator instances with class-weight handling where supported."""
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=8,
            min_samples_leaf=10,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
            C=1.0,
            gamma="scale",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
        ),
    }
    return models


def train_and_compare(splits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fit each model on transformed training data, score on validation,
    optionally run a SMOTE experiment for comparison, then evaluate the
    selected model once on the test set.
    """
    ensure_directories()

    X_train_t = splits["X_train_t"]
    X_val_t = splits["X_val_t"]
    X_test_t = splits["X_test_t"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]
    y_test = splits["y_test"]
    preprocessor = splits["preprocessor"]
    feature_names = splits["feature_names"]

    results: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Any] = {}

    candidates = build_candidate_models()

    print("Training candidate models...")
    for name, estimator in candidates.items():
        print(f"  -> {name}")
        model = estimator
        # Gradient Boosting has no class_weight; use sample_weight instead
        if name == "Gradient Boosting":
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train_t, y_train, sample_weight=sw)
        else:
            model.fit(X_train_t, y_train)

        # Cross-validation F1 on training data only
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if name == "Gradient Boosting":
            # sample_weight not easily passed via cross_val_score for all folds;
            # report train-set CV without sample weights for transparency
            cv_scores = cross_val_score(
                estimator.__class__(**estimator.get_params()),
                X_train_t,
                y_train,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
            )
        else:
            cv_scores = cross_val_score(
                estimator.__class__(**estimator.get_params()),
                X_train_t,
                y_train,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
            )

        y_val_pred = model.predict(X_val_t)
        y_val_proba = _get_proba(model, X_val_t)
        val_metrics = _metrics(y_val, y_val_pred, y_val_proba)
        val_metrics["cv_f1_mean"] = float(np.mean(cv_scores))
        val_metrics["cv_f1_std"] = float(np.std(cv_scores))

        results[name] = {
            "validation": val_metrics,
            "selection_metric": "validation_f1",
        }
        fitted_models[name] = model

    # Optional SMOTE experiment on Logistic Regression (train only)
    smote_note = "imbalanced-learn not available; SMOTE experiment skipped."
    if HAS_IMBLEARN:
        print("  -> Logistic Regression + SMOTE (train-only experiment)")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train_t, y_train)
        smote_model = LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        )
        smote_model.fit(X_res, y_res)
        y_val_pred = smote_model.predict(X_val_t)
        y_val_proba = _get_proba(smote_model, X_val_t)
        val_metrics = _metrics(y_val, y_val_pred, y_val_proba)
        results["Logistic Regression + SMOTE"] = {
            "validation": val_metrics,
            "selection_metric": "validation_f1",
            "note": (
                "SMOTE applied only to training data after the stratified split. "
                "Compared against class-weight baseline; not assumed to be better."
            ),
        }
        fitted_models["Logistic Regression + SMOTE"] = smote_model
        smote_note = (
            "SMOTE experiment completed on training data only and compared on validation F1."
        )

    # Select best by validation F1 (not test)
    best_name = max(
        results.keys(),
        key=lambda n: results[n]["validation"]["f1"],
    )
    best_model = fitted_models[best_name]
    print(f"Selected model (by validation F1): {best_name}")

    # Final evaluation on untouched test set — once
    y_test_pred = best_model.predict(X_test_t)
    y_test_proba = _get_proba(best_model, X_test_t)
    test_metrics = _metrics(y_test, y_test_pred, y_test_proba)

    # Also compute test metrics for all models for comparison tables
    # (reported separately; selection was already locked using validation)
    comparison_rows: List[Dict[str, Any]] = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test_t)
        y_proba = _get_proba(model, X_test_t)
        test_m = _metrics(y_test, y_pred, y_proba)
        results[name]["test"] = test_m
        comparison_rows.append(
            {
                "model": name,
                "val_accuracy": results[name]["validation"]["accuracy"],
                "val_precision": results[name]["validation"]["precision"],
                "val_recall": results[name]["validation"]["recall"],
                "val_f1": results[name]["validation"]["f1"],
                "val_roc_auc": results[name]["validation"]["roc_auc"],
                "test_accuracy": test_m["accuracy"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_f1": test_m["f1"],
                "test_roc_auc": test_m["roc_auc"],
                "selected": name == best_name,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("val_f1", ascending=False)
    comparison_path = MODELS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Full inference pipeline: raw features -> preprocessor -> classifier
    inference_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_model),
        ]
    )
    # Re-fit classifier already fitted on train transform; preprocessor already fitted.
    # Rebuild cleanly so joblib dump stores both stages consistently.
    # The classifier was trained on X_train_t from this same preprocessor.
    model_path = MODELS_DIR / "trained_model.joblib"
    joblib.dump(inference_pipeline, model_path)

    # Also store the selected classifier alone for evaluation plots
    joblib.dump(best_model, MODELS_DIR / "best_classifier.joblib")
    joblib.dump(fitted_models, MODELS_DIR / "all_fitted_models.joblib")

    metadata = {
        "selected_model": best_name,
        "selection_rule": (
            "Highest validation-set F1-score for the positive (stroke) class. "
            "Test-set metrics were computed after selection and are not used to choose the model."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "encoded_feature_names": feature_names,
        "random_state": RANDOM_STATE,
        "smote_experiment": smote_note,
        "test_metrics_selected": test_metrics,
        "validation_metrics_selected": results[best_name]["validation"],
        "model_path": str(model_path),
        "medical_disclaimer": (
            "Educational prototype only. Predictions are not diagnoses. "
            "Scores are model outputs, not clinically calibrated probabilities of future stroke."
        ),
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_payload = {
        "results": results,
        "selected_model": best_name,
        "comparison_table": comparison_rows,
        "test_metrics_selected": test_metrics,
    }
    with open(OUTPUTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return {
        "results": results,
        "selected_model": best_name,
        "best_model": best_model,
        "fitted_models": fitted_models,
        "inference_pipeline": inference_pipeline,
        "comparison_df": comparison_df,
        "metadata": metadata,
        "splits": splits,
    }


def run_training() -> Dict[str, Any]:
    print("=" * 60)
    print("Brain StrokeSense — Model Training")
    print("=" * 60)
    splits = prepare_splits()
    outcome = train_and_compare(splits)
    print(f"Best model: {outcome['selected_model']}")
    print(f"Test F1: {outcome['metadata']['test_metrics_selected']['f1']:.4f}")
    print(f"Artifacts saved under {MODELS_DIR} and {OUTPUTS_DIR}")
    return outcome


if __name__ == "__main__":
    run_training()
