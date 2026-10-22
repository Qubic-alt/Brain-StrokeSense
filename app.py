"""
Brain StrokeSense — Flask Web Application
=========================================
Serves the multi-page healthcare AI dashboard for dataset exploration,
EDA, model comparison, prediction demo, and explainability.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from preprocess import (
    BASE_DIR,
    FEATURE_COLUMNS,
    GRAPHS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    DATASET_PATH,
    ensure_directories,
    load_dataset,
)

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "brainstrokesense-college-demo-key"
app.config["TEMPLATES_AUTO_RELOAD"] = True

ensure_directories()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}.") from exc


def get_key_stats() -> Dict[str, Any]:
    """Dataset / model stats for the home page (real values when available)."""
    stats: Dict[str, Any] = {
        "n_rows": "—",
        "n_features": len(FEATURE_COLUMNS),
        "stroke_rate": "—",
        "selected_model": "Not trained yet",
        "test_f1": "—",
        "dataset_ready": DATASET_PATH.exists(),
        "model_ready": (MODELS_DIR / "trained_model.joblib").exists(),
        "graphs_ready": any(GRAPHS_DIR.glob("*.png")),
    }
    overview = _load_json(OUTPUTS_DIR / "dataset_summary.json")
    if overview:
        stats["n_rows"] = overview.get("shape", {}).get("rows", "—")
        td = overview.get("target_percentages", {})
        if "1" in td:
            stats["stroke_rate"] = f"{td['1']}%"
    meta = _load_json(MODELS_DIR / "model_metadata.json")
    if meta:
        stats["selected_model"] = meta.get("selected_model", "—")
        tm = meta.get("test_metrics_selected", {})
        if tm.get("f1") is not None:
            stats["test_f1"] = f"{tm['f1']:.3f}"
    return stats


def load_inference_pipeline():
    path = MODELS_DIR / "trained_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", stats=get_key_stats())


@app.route("/dataset")
def dataset_page():
    error = None
    overview = _load_json(OUTPUTS_DIR / "dataset_summary.json")
    preview_rows = []
    columns = []
    try:
        df = load_dataset()
        if overview is None:
            from preprocess import get_dataset_overview

            overview = get_dataset_overview(df)
        columns = list(df.columns)
        # Simple pagination
        page = max(1, int(request.args.get("page", 1)))
        per_page = 15
        start = (page - 1) * per_page
        end = start + per_page
        preview_rows = (
            df.iloc[start:end]
            .fillna("")
            .astype(object)
            .to_dict(orient="records")
        )
        total_pages = max(1, (len(df) + per_page - 1) // per_page)
    except Exception as exc:
        error = str(exc)
        page, total_pages = 1, 1

    return render_template(
        "dataset.html",
        overview=overview,
        preview_rows=preview_rows,
        columns=columns,
        page=page,
        total_pages=total_pages,
        error=error,
    )


@app.route("/preprocessing")
def preprocessing_page():
    summary = _load_json(OUTPUTS_DIR / "preprocessing_summary.json")
    overview = _load_json(OUTPUTS_DIR / "dataset_summary.json")
    graphs = [
        "missing_values.png",
        "before_after_dedup.png",
        "dtypes_summary.png",
        "correlation_heatmap.png",
    ]
    available = [g for g in graphs if (GRAPHS_DIR / g).exists()]
    return render_template(
        "preprocessing.html",
        summary=summary,
        overview=overview,
        graphs=available,
    )


@app.route("/analysis")
def analysis_page():
    catalog = _load_json(OUTPUTS_DIR / "eda_catalog.json")
    charts = []
    if catalog:
        charts = [c for c in catalog.get("charts", []) if c.get("file")]
        # Keep only files that exist
        charts = [c for c in charts if (GRAPHS_DIR / c["file"]).exists()]
    filter_type = request.args.get("type", "all")
    if filter_type != "all":
        charts = [c for c in charts if c.get("type") == filter_type]
    types = sorted({c.get("type", "other") for c in (catalog or {}).get("charts", []) if c.get("type")})
    return render_template(
        "analysis.html",
        charts=charts,
        catalog=catalog,
        filter_type=filter_type,
        types=types,
    )


@app.route("/models")
def models_page():
    metrics = _load_json(OUTPUTS_DIR / "model_metrics.json")
    meta = _load_json(MODELS_DIR / "model_metadata.json")
    eval_catalog = _load_json(OUTPUTS_DIR / "evaluation_catalog.json")
    comparison = None
    cmp_path = MODELS_DIR / "model_comparison.csv"
    if cmp_path.exists():
        comparison = pd.read_csv(cmp_path).to_dict(orient="records")

    chart_files = [
        "model_comparison_bars.png",
        "roc_curves.png",
        "pr_curves.png",
        "confusion_test_selected.png",
    ]
    # Include per-model confusion matrices
    if GRAPHS_DIR.exists():
        chart_files.extend(
            sorted(p.name for p in GRAPHS_DIR.glob("confusion_*.png") if p.name not in chart_files)
        )

    charts = []
    interpretations = {
        c["file"]: c.get("interpretation", "")
        for c in (eval_catalog or {}).get("charts", [])
        if c.get("file")
    }
    for f in chart_files:
        if (GRAPHS_DIR / f).exists():
            charts.append({"file": f, "interpretation": interpretations.get(f, "")})

    return render_template(
        "models.html",
        metrics=metrics,
        meta=meta,
        comparison=comparison,
        charts=charts,
        eval_catalog=eval_catalog,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    result = None
    error = None
    form_data = {
        "gender": "Female",
        "age": "45",
        "hypertension": "0",
        "heart_disease": "0",
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": "95.0",
        "bmi": "28.0",
        "smoking_status": "never smoked",
    }

    if request.method == "POST":
        form_data = {k: request.form.get(k, form_data.get(k, "")) for k in form_data}
        try:
            pipeline = load_inference_pipeline()
            if pipeline is None:
                raise FileNotFoundError(
                    "Trained model not found. Run: python evaluate_models.py"
                )

            age = _safe_float(form_data["age"], "age")
            glucose = _safe_float(form_data["avg_glucose_level"], "avg_glucose_level")
            bmi_raw = form_data.get("bmi", "").strip()
            if bmi_raw == "":
                bmi = float("nan")  # handled by imputer
            else:
                bmi = _safe_float(bmi_raw, "bmi")

            if not (0 < age < 120):
                raise ValueError("Age must be between 0 and 120.")
            if not (0 < glucose < 500):
                raise ValueError("Average glucose level looks out of range.")
            if bmi == bmi and not (10 < bmi < 100):  # nan-safe check
                raise ValueError("BMI looks out of range (or leave blank to impute).")

            row = {
                "gender": form_data["gender"],
                "age": age,
                "hypertension": int(form_data["hypertension"]),
                "heart_disease": int(form_data["heart_disease"]),
                "ever_married": form_data["ever_married"],
                "work_type": form_data["work_type"],
                "Residence_type": form_data["Residence_type"],
                "avg_glucose_level": glucose,
                "bmi": bmi,
                "smoking_status": form_data["smoking_status"],
            }
            X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
            pred = int(pipeline.predict(X)[0])
            score = None
            if hasattr(pipeline, "predict_proba"):
                score = float(pipeline.predict_proba(X)[0, 1])
            elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
                # Not a calibrated probability
                raw = float(pipeline.decision_function(X)[0])
                score = raw

            meta = _load_json(MODELS_DIR / "model_metadata.json") or {}
            result = {
                "label": "Stroke indicated by model" if pred == 1 else "No stroke indicated by model",
                "class": pred,
                "score": score,
                "model_name": meta.get("selected_model", "Selected pipeline"),
                "bmi_imputed": bmi_raw == "",
            }
        except Exception as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        result=result,
        error=error,
        form_data=form_data,
        model_ready=(MODELS_DIR / "trained_model.joblib").exists(),
    )


@app.route("/explainability")
def explainability_page():
    importance = _load_json(OUTPUTS_DIR / "feature_importance.json")
    meta = _load_json(MODELS_DIR / "model_metadata.json")
    chart_exists = (GRAPHS_DIR / "feature_importance.png").exists()
    return render_template(
        "explainability.html",
        importance=importance,
        meta=meta,
        chart_exists=chart_exists,
    )


@app.route("/research")
def research_page():
    return render_template("research.html", stats=get_key_stats())


@app.route("/outputs/graphs/<path:filename>")
def graph_file(filename: str):
    return send_from_directory(GRAPHS_DIR, filename)


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", stats=get_key_stats()), 404


if __name__ == "__main__":
    print("=" * 60)
    print("Brain StrokeSense — Flask Server")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    if not DATASET_PATH.exists():
        print("WARNING: Dataset CSV missing in data/")
    if not (MODELS_DIR / "trained_model.joblib").exists():
        print("WARNING: Model not trained. Run: python evaluate_models.py")
    app.run(host="127.0.0.1", port=5000, debug=False)
