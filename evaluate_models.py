"""
Brain StrokeSense — Evaluation & Explainability
===============================================
Creates comparison charts, confusion matrices, ROC / PR curves,
permutation importance, and persists artifacts for the Flask dashboard.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from preprocess import (
    GRAPHS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    ensure_directories,
    prepare_splits,
)
from train_model import run_training, _get_proba

warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": "#0b1220",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "text.color": "#e2e8f0",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "grid.color": "#1f2937",
        "savefig.facecolor": "#0b1220",
    }
)

TEAL = "#14b8a6"
CYAN = "#22d3ee"
CORAL = "#f87171"
AMBER = "#fbbf24"
BLUE = "#38bdf8"
PALETTE = [TEAL, CYAN, CORAL, AMBER, BLUE, "#a78bfa", "#34d399"]


def _save(fig: plt.Figure, filename: str) -> str:
    ensure_directories()
    path = GRAPHS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_comparison_bars(comparison_df: pd.DataFrame) -> List[Dict[str, Any]]:
    charts = []
    metrics = [
        ("val_f1", "Validation F1"),
        ("val_recall", "Validation Recall"),
        ("val_precision", "Validation Precision"),
        ("val_roc_auc", "Validation ROC-AUC"),
        ("val_accuracy", "Validation Accuracy"),
        
def plot_validation_metric_charts(comparison_df):
    """Create a separate validation comparison chart for each metric."""
    charts = []

    metrics = [
        ("val_accuracy", "Validation Accuracy", "validation_accuracy_comparison.png"),
        ("val_precision", "Validation Precision", "validation_precision_comparison.png"),
        ("val_recall", "Validation Recall", "validation_recall_comparison.png"),
        ("val_f1", "Validation F1-score", "validation_f1_comparison.png"),
        ("val_roc_auc", "Validation ROC-AUC", "validation_roc_auc_comparison.png"),
    ]

    for column, title, filename in metrics:
        if column not in comparison_df.columns:
            continue

        plot_df = comparison_df[["model", column]].copy()
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df = plot_df.dropna(subset=[column])

        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))

        bars = ax.bar(
            plot_df["model"],
            plot_df[column],
            color=PALETTE[:len(plot_df)] if len(plot_df) <= len(PALETTE)
            else [PALETTE[i % len(PALETTE)] for i in range(len(plot_df))]
        )

        ax.set_title(f"{title} — All Models")
        ax.set_xlabel("Machine Learning Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=25)

        for bar, score in zip(bars, plot_df[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        saved_file = _save(fig, filename)

        charts.append({
            "file": saved_file,
            "title": title + " — Model Comparison",
            "interpretation": (
                f"Compares {title.lower()} across all evaluated models "
                "using the validation dataset. Higher scores indicate "
                "better performance for this metric. These results are "
                "for model comparison, not clinical diagnosis."
            ),
        })

    return charts

    ]
    plot_df = comparison_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(plot_df))
    width = 0.15
    for i, (col, label) in enumerate(metrics):
        vals = plot_df[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=label, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(plot_df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (Validation Metrics)")
    ax.legend(loc="upper right", fontsize=8)
    fname = _save(fig, "model_comparison_bars.png")
    charts.append(
        {
            "file": fname,
            "title": "Validation metric comparison",
            "interpretation": (
                "Side-by-side validation metrics. Model selection uses validation F1 so "
                "accuracy is not treated as the primary objective under class imbalance."
            ),
        }
    )
    return charts


def plot_confusion_matrices(fitted_models: Dict[str, Any], X_val, y_val) -> List[Dict[str, Any]]:
    charts = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="mako",
            ax=ax,
            xticklabels=["No Stroke", "Stroke"],
            yticklabels=["No Stroke", "Stroke"],
        )
        safe = name.replace(" ", "_").replace("+", "plus")
        ax.set_title(f"Confusion Matrix — {name} (Validation)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fname = _save(fig, f"confusion_{safe}.png")
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        charts.append(
            {
                "file": fname,
                "title": f"Confusion matrix: {name}",
                "interpretation": (
                    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}. False negatives (missed stroke cases) "
                    "are especially concerning in screening-style settings; false positives "
                    "may cause unnecessary follow-up. This is an educational model only."
                ),
            }
        )
    return charts


def plot_roc_curves(fitted_models: Dict[str, Any], X_val, y_val) -> List[Dict[str, Any]]:
    charts = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, model) in enumerate(fitted_models.items()):
        proba = _get_proba(model, X_val)
        try:
            RocCurveDisplay.from_predictions(y_val, proba, name=name, ax=ax)
        except Exception:
            fpr, tpr, _ = roc_curve(y_val, proba)
            ax.plot(fpr, tpr, label=name, color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Chance")
    ax.set_title("ROC Curves (Validation)")
    ax.legend(loc="lower right", fontsize=8)
    fname = _save(fig, "roc_curves.png")
    charts.append(
        {
            "file": fname,
            "title": "ROC curves",
            "interpretation": (
                "ROC curves show ranking quality across thresholds. AUC near 0.5 is chance. "
                "With imbalance, also inspect precision-recall curves."
            ),
        }
    )
    return charts


def plot_pr_curves(fitted_models: Dict[str, Any], X_val, y_val) -> List[Dict[str, Any]]:
    charts = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, model) in enumerate(fitted_models.items()):
        proba = _get_proba(model, X_val)
        try:
            PrecisionRecallDisplay.from_predictions(y_val, proba, name=name, ax=ax)
        except Exception:
            precision, recall, _ = precision_recall_curve(y_val, proba)
            ax.plot(recall, precision, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_title("Precision–Recall Curves (Validation)")
    ax.legend(loc="upper right", fontsize=8)
    fname = _save(fig, "pr_curves.png")
    charts.append(
        {
            "file": fname,
            "title": "Precision-recall curves",
            "interpretation": (
                "Precision-recall curves are informative under class imbalance. "
                "Average precision summarizes the curve; high recall often reduces precision."
            ),
        }
    )
    return charts


def compute_permutation_importance(
    best_model,
    X_val_t,
    y_val,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Permutation importance on validation data for the selected model."""
    result = permutation_importance(
        best_model,
        X_val_t,
        y_val,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=-1,
    )
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp_path = OUTPUTS_DIR / "feature_importance.csv"
    importances.to_csv(imp_path, index=False)

    top = importances.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"][::-1],
        top["importance_mean"][::-1],
        xerr=top["importance_std"][::-1],
        color=TEAL,
        ecolor=CYAN,
    )
    ax.set_title("Permutation Feature Importance (Validation F1)")
    ax.set_xlabel("Mean importance (F1 drop when shuffled)")
    fname = _save(fig, "feature_importance.png")

    payload = {
        "method": "sklearn.inspection.permutation_importance",
        "scoring": "f1",
        "n_repeats": 10,
        "chart": fname,
        "top_features": top.to_dict(orient="records"),
        "all_features": importances.to_dict(orient="records"),
        "caveats": [
            "Importance measures association with model predictions, not biological causation.",
            "Correlated features can share or dilute importance.",
            "Encoded categorical columns appear as one-hot feature names.",
            "This does not explain the medical cause of stroke.",
        ],
    }
    with open(OUTPUTS_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def plot_test_confusion(best_name: str, best_model, X_test, y_test) -> List[Dict[str, Any]]:
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="rocket",
        ax=ax,
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
    )
    ax.set_title(f"Final Test Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fname = _save(fig, "confusion_test_selected.png")
    return [
        {
            "file": fname,
            "title": "Selected model — test confusion matrix",
            "interpretation": (
                "Untouched test-set confusion matrix for the model selected by validation F1. "
                "This is a single final estimate and was not used for model choice."
            ),
        }
    ]


def run_evaluation(train_outcome: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_directories()
    if train_outcome is None:
        # Prefer loading existing artifacts; otherwise train
        metrics_path = OUTPUTS_DIR / "model_metrics.json"
        models_path = MODELS_DIR / "all_fitted_models.joblib"
        if metrics_path.exists() and models_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            fitted = joblib.load(models_path)
            splits = prepare_splits()
            best_name = metrics["selected_model"]
            best_model = fitted[best_name]
            comparison_df = pd.read_csv(MODELS_DIR / "model_comparison.csv")
            feature_names = splits["feature_names"]
        else:
            train_outcome = run_training()
            fitted = train_outcome["fitted_models"]
            best_name = train_outcome["selected_model"]
            best_model = train_outcome["best_model"]
            comparison_df = train_outcome["comparison_df"]
            splits = train_outcome["splits"]
            feature_names = splits["feature_names"]
            metrics = {
                "results": train_outcome["results"],
                "selected_model": best_name,
            }
    else:
        fitted = train_outcome["fitted_models"]
        best_name = train_outcome["selected_model"]
        best_model = train_outcome["best_model"]
        comparison_df = train_outcome["comparison_df"]
        splits = train_outcome["splits"]
        feature_names = splits["feature_names"]
        metrics = {
            "results": train_outcome["results"],
            "selected_model": best_name,
        }

    charts: List[Dict[str, Any]] = []
    charts.extend(plot_comparison_bars(comparison_df))
    charts.extend(plot_confusion_matrices(fitted, splits["X_val_t"], splits["y_val"]))
    charts.extend(plot_roc_curves(fitted, splits["X_val_t"], splits["y_val"]))
    charts.extend(plot_pr_curves(fitted, splits["X_val_t"], splits["y_val"]))
    charts.extend(plot_test_confusion(best_name, best_model, splits["X_test_t"], splits["y_test"]))

    importance = compute_permutation_importance(
        best_model, splits["X_val_t"], splits["y_val"], feature_names
    )
    charts.append(
        {
            "file": importance["chart"],
            "title": "Permutation feature importance",
            "interpretation": (
                "Features that most reduce validation F1 when randomly shuffled are ranked higher. "
                "Importance ≠ causation. See explainability page for caveats."
            ),
        }
    )

    catalog = {
        "selected_model": best_name,
        "charts": charts,
        "accuracy_warning": (
            "Accuracy can be misleading when stroke cases are rare: a model that always "
            "predicts 'no stroke' may achieve high accuracy while failing to identify the "
            "minority class. Prefer recall, precision, F1, ROC-AUC, and confusion matrices."
        ),
    }
    with open(OUTPUTS_DIR / "evaluation_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Evaluation charts saved to {GRAPHS_DIR}")
    return {"catalog": catalog, "importance": importance, "metrics": metrics}


if __name__ == "__main__":
    print("=" * 60)
    print("Brain StrokeSense — Evaluation & Explainability")
    print("=" * 60)
    # Train then evaluate in one go for first-time setup
    outcome = run_training()
    run_evaluation(outcome)
    print("Evaluation complete.")
