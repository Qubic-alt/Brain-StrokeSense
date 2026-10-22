"""
Brain StrokeSense — Exploratory Data Analysis
=============================================
Generates graphs and interpretations for every dataset attribute,
plus dataset-quality analyses. Charts are saved under outputs/graphs/
so they persist across Flask restarts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocess import (
    BASE_DIR,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    GRAPHS_DIR,
    NUMERICAL_FEATURES,
    OUTPUTS_DIR,
    TARGET_COLUMN,
    clean_dataframe,
    ensure_directories,
    get_dataset_overview,
    load_dataset,
    prepare_splits,
)

warnings.filterwarnings("ignore")

# Consistent visual theme matching the website
sns.set_theme(style="darkgrid", context="notebook")
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
        "font.size": 11,
    }
)

TEAL = "#14b8a6"
CYAN = "#22d3ee"
CORAL = "#f87171"
AMBER = "#fbbf24"
BLUE = "#38bdf8"
PALETTE = [TEAL, CYAN, CORAL, AMBER, BLUE, "#a78bfa"]


def _save(fig: plt.Figure, filename: str) -> str:
    ensure_directories()
    path = GRAPHS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return filename


def _stroke_label(series: pd.Series) -> pd.Series:
    return series.map({0: "No Stroke", 1: "Stroke"}).fillna("Unknown")


def analyze_numerical(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Histograms, box plots, KDE, and stroke comparisons for numerical attrs."""
    charts: List[Dict[str, Any]] = []

    for col in NUMERICAL_FEATURES:
        series = df[col]
        missing = int(series.isna().sum())
        desc = series.describe().to_dict()

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(series.dropna(), bins=30, color=TEAL, edgecolor="#0f172a", alpha=0.9)
        ax.set_title(f"Histogram — {col}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        fname = _save(fig, f"hist_{col}.png")
        charts.append(
            {
                "file": fname,
                "title": f"Histogram of {col}",
                "attribute": col,
                "type": "numerical",
                "interpretation": (
                    f"{col} ranges from {desc.get('min', float('nan')):.2f} to "
                    f"{desc.get('max', float('nan')):.2f} (mean ≈ {desc.get('mean', float('nan')):.2f}). "
                    f"Missing values: {missing}. The histogram shows how values are distributed "
                    f"across patients in the dataset."
                ),
            }
        )

        # Box plot
        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.boxplot(x=series, ax=ax, color=CYAN)
        ax.set_title(f"Box Plot — {col} (outlier inspection)")
        ax.set_xlabel(col.replace("_", " ").title())
        fname = _save(fig, f"box_{col}.png")
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_out = int(
            ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        )
        charts.append(
            {
                "file": fname,
                "title": f"Box plot of {col}",
                "attribute": col,
                "type": "numerical",
                "interpretation": (
                    f"IQR-based inspection flags about {n_out} potential outliers for {col}. "
                    "These points are retained because extreme health measurements can be "
                    "clinically legitimate and removing them without domain justification "
                    "could bias the analysis."
                ),
            }
        )

        # Distribution / KDE
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.kdeplot(series.dropna(), ax=ax, fill=True, color=BLUE, linewidth=2)
        ax.set_title(f"Density Plot — {col}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Density")
        fname = _save(fig, f"kde_{col}.png")
        charts.append(
            {
                "file": fname,
                "title": f"Density of {col}",
                "attribute": col,
                "type": "numerical",
                "interpretation": (
                    f"The smoothed density estimate for {col} highlights modality and skewness, "
                    "which informs whether scaling and robust imputation are appropriate."
                ),
            }
        )

        # Comparison vs stroke
        plot_df = df[[col, TARGET_COLUMN]].dropna().copy()
        plot_df["stroke_label"] = _stroke_label(plot_df[TARGET_COLUMN])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=plot_df,
            x="stroke_label",
            y=col,
            palette=[TEAL, CORAL],
            ax=ax,
        )
        ax.set_title(f"{col} by Stroke Outcome")
        ax.set_xlabel("Stroke Outcome")
        ax.set_ylabel(col.replace("_", " ").title())
        fname = _save(fig, f"box_stroke_{col}.png")
        charts.append(
            {
                "file": fname,
                "title": f"{col} vs stroke",
                "attribute": col,
                "type": "numerical",
                "interpretation": (
                    f"Comparing {col} across stroke vs non-stroke groups shows whether "
                    "central tendency and spread differ between recorded outcomes. "
                    "Differences suggest association in this dataset, not causation."
                ),
            }
        )

        # Overlay histogram by stroke
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, color in [("No Stroke", TEAL), ("Stroke", CORAL)]:
            subset = plot_df.loc[plot_df["stroke_label"] == label, col]
            ax.hist(
                subset,
                bins=25,
                alpha=0.55,
                label=label,
                color=color,
                edgecolor="#0f172a",
            )
        ax.set_title(f"{col} Distribution by Stroke Class")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend()
        fname = _save(fig, f"hist_stroke_{col}.png")
        charts.append(
            {
                "file": fname,
                "title": f"{col} histogram by stroke",
                "attribute": col,
                "type": "numerical",
                "interpretation": (
                    f"Overlaying class-wise histograms for {col} helps visualize overlap "
                    "between stroke and non-stroke groups — high overlap implies weaker "
                    "separability for that single feature alone."
                ),
            }
        )

    # Correlation heatmap for numerical + target
    corr_cols = [c for c in NUMERICAL_FEATURES + BINARY_FEATURES + [TARGET_COLUMN] if c in df.columns]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.4,
        linecolor="#1f2937",
    )
    ax.set_title("Correlation Heatmap (Numerical / Binary / Target)")
    fname = _save(fig, "correlation_heatmap.png")
    charts.append(
        {
            "file": fname,
            "title": "Numerical correlation heatmap",
            "attribute": "correlation",
            "type": "quality",
            "interpretation": (
                "Pearson correlations among numerical/binary features and the stroke target. "
                "Strong correlation with stroke may indicate predictive association in this "
                "sample, but correlation is not causation and can be distorted by imbalance."
            ),
        }
    )

    return charts


def analyze_categorical(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Count plots, percentages, and stroke crosstabs for categorical/binary attrs."""
    charts: List[Dict[str, Any]] = []
    cat_cols = CATEGORICAL_FEATURES + BINARY_FEATURES

    for col in cat_cols:
        if col not in df.columns:
            charts.append(
                {
                    "file": None,
                    "title": f"{col} unavailable",
                    "attribute": col,
                    "type": "categorical",
                    "interpretation": (
                        f"Attribute '{col}' is not present in the loaded CSV, so no chart "
                        "was generated for it."
                    ),
                }
            )
            continue

        counts = df[col].fillna("Missing").astype(str).value_counts()
        pct = (100.0 * counts / counts.sum()).round(2)

        # Count plot
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = counts.index.tolist()
        sns.countplot(
            data=df.assign(**{col: df[col].fillna("Missing").astype(str)}),
            x=col,
            order=order,
            ax=ax,
            palette=PALETTE,
        )
        ax.set_title(f"Count Plot — {col}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fname = _save(fig, f"count_{col}.png")
        top = counts.index[0]
        charts.append(
            {
                "file": fname,
                "title": f"Counts for {col}",
                "attribute": col,
                "type": "categorical",
                "interpretation": (
                    f"Category frequencies for {col}. Most common value: '{top}' "
                    f"({int(counts.iloc[0])} rows, {pct.iloc[0]}%). "
                    f"Full distribution: "
                    + ", ".join(f"{i}={int(c)} ({pct[i]}%)" for i, c in counts.items())
                    + "."
                ),
            }
        )

        # Percentage bar
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pct.plot(kind="bar", ax=ax, color=CYAN, edgecolor="#0f172a")
        ax.set_title(f"Percentage Distribution — {col}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Percentage (%)")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fname = _save(fig, f"pct_{col}.png")
        charts.append(
            {
                "file": fname,
                "title": f"Percentages for {col}",
                "attribute": col,
                "type": "categorical",
                "interpretation": (
                    f"Relative percentages for each {col} category, useful for spotting "
                    "rare levels that may be unstable during stratified splitting."
                ),
            }
        )

        # Stroke comparison (grouped counts)
        plot_df = df[[col, TARGET_COLUMN]].copy()
        plot_df[col] = plot_df[col].fillna("Missing").astype(str)
        plot_df["stroke_label"] = _stroke_label(plot_df[TARGET_COLUMN])
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.countplot(
            data=plot_df,
            x=col,
            hue="stroke_label",
            order=order,
            palette=[TEAL, CORAL],
            ax=ax,
        )
        ax.set_title(f"{col} by Stroke Outcome")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend(title="Outcome")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fname = _save(fig, f"count_stroke_{col}.png")

        # Stroke rate per category for interpretation
        rate = (
            plot_df.groupby(col)[TARGET_COLUMN]
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        )
        rate_txt = ", ".join(f"{k}: {v}%" for k, v in rate.items())
        charts.append(
            {
                "file": fname,
                "title": f"{col} vs stroke",
                "attribute": col,
                "type": "categorical",
                "interpretation": (
                    f"Stroke vs non-stroke counts within each {col} category. "
                    f"Observed stroke rate by category: {rate_txt}. "
                    "Rates are descriptive associations in this dataset only."
                ),
            }
        )

    return charts


def analyze_target(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Target class distribution charts and imbalance explanation."""
    charts: List[Dict[str, Any]] = []
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No Stroke (0)", "Stroke (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    total = sum(values) or 1
    pct = [round(100.0 * v / total, 2) for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=[TEAL, CORAL], edgecolor="#0f172a")
    ax.set_title("Stroke vs Non-Stroke Class Distribution")
    ax.set_ylabel("Count")
    for bar, p in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}\n({p}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fname = _save(fig, "target_distribution.png")
    charts.append(
        {
            "file": fname,
            "title": "Target class distribution",
            "attribute": TARGET_COLUMN,
            "type": "target",
            "interpretation": (
                f"Class counts: No Stroke={values[0]} ({pct[0]}%), "
                f"Stroke={values[1]} ({pct[1]}%). "
                "The minority stroke class means accuracy alone can look high while "
                "missing many true stroke cases — hence F1, recall, and ROC-AUC are "
                "emphasized during model selection."
            ),
        }
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=[TEAL, CORAL],
        startangle=90,
        wedgeprops={"edgecolor": "#0f172a"},
        textprops={"color": "#e2e8f0"},
    )
    ax.set_title("Stroke Class Share")
    fname = _save(fig, "target_pie.png")
    charts.append(
        {
            "file": fname,
            "title": "Target class pie chart",
            "attribute": TARGET_COLUMN,
            "type": "target",
            "interpretation": (
                "Pie chart of class share. Severe imbalance is expected in stroke "
                "screening datasets and motivates class weights / careful metric choice."
            ),
        }
    )
    return charts


def analyze_id_overview(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Include id only in overview — uniqueness check, not predictive analysis."""
    charts: List[Dict[str, Any]] = []
    if "id" not in df.columns:
        return charts

    n_unique = int(df["id"].nunique())
    n_rows = int(len(df))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(
        ["Unique IDs", "Total Rows"],
        [n_unique, n_rows],
        color=[CYAN, TEAL],
        edgecolor="#0f172a",
    )
    ax.set_title("Identifier Column Overview (id)")
    ax.set_ylabel("Count")
    fname = _save(fig, "id_overview.png")
    charts.append(
        {
            "file": fname,
            "title": "ID column overview",
            "attribute": "id",
            "type": "overview",
            "interpretation": (
                f"The id column has {n_unique} unique values across {n_rows} rows. "
                "It is excluded from predictive features because identifiers do not "
                "generalize and can cause leakage-like memorization."
            ),
        }
    )
    return charts


def analyze_quality(df: pd.DataFrame, cleaned: pd.DataFrame) -> List[Dict[str, Any]]:
    """Missing values, duplicates, dtypes, before/after preprocessing summaries."""
    charts: List[Dict[str, Any]] = []

    # Missing values by column
    missing = df.isna().sum()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    missing.plot(kind="bar", ax=ax, color=AMBER, edgecolor="#0f172a")
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fname = _save(fig, "missing_values.png")
    miss_txt = ", ".join(f"{c}={int(v)}" for c, v in missing.items() if v > 0) or "none"
    charts.append(
        {
            "file": fname,
            "title": "Missing values by column",
            "attribute": "quality",
            "type": "quality",
            "interpretation": (
                f"Missing-value counts before imputation. Non-zero columns: {miss_txt}. "
                "BMI missingness is handled with median imputation fitted on training data only."
            ),
        }
    )

    # Dtype summary bar
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    dtype_counts.plot(kind="bar", ax=ax, color=BLUE, edgecolor="#0f172a")
    ax.set_title("Column Data Types")
    ax.set_xlabel("dtype")
    ax.set_ylabel("Number of Columns")
    fname = _save(fig, "dtypes_summary.png")
    charts.append(
        {
            "file": fname,
            "title": "Data types summary",
            "attribute": "quality",
            "type": "quality",
            "interpretation": (
                "Count of pandas dtypes in the raw CSV. Object columns are treated as "
                "categorical features and one-hot encoded after train-only fitting."
            ),
        }
    )

    # Before vs after row counts / duplicates
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Raw Rows", "After Dedup", "Duplicates Removed"],
        [
            len(df),
            len(cleaned),
            int(df.duplicated().sum()),
        ],
        color=[TEAL, CYAN, CORAL],
        edgecolor="#0f172a",
    )
    ax.set_title("Before vs After Deduplication")
    ax.set_ylabel("Count")
    fname = _save(fig, "before_after_dedup.png")
    charts.append(
        {
            "file": fname,
            "title": "Before/after duplicate handling",
            "attribute": "quality",
            "type": "quality",
            "interpretation": (
                f"Raw rows={len(df)}, after dropping exact duplicates={len(cleaned)}, "
                f"duplicates removed={int(df.duplicated().sum())}."
            ),
        }
    )

    # Pairplot-lite: scatter age vs glucose colored by stroke
    sample = df[["age", "avg_glucose_level", TARGET_COLUMN]].dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [(0, TEAL), (1, CORAL)]:
        part = sample[sample[TARGET_COLUMN] == label]
        ax.scatter(
            part["age"],
            part["avg_glucose_level"],
            s=18,
            alpha=0.55,
            c=color,
            label="Stroke" if label == 1 else "No Stroke",
            edgecolors="none",
        )
    ax.set_title("Age vs Average Glucose Level by Stroke")
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Glucose Level")
    ax.legend()
    fname = _save(fig, "scatter_age_glucose.png")
    charts.append(
        {
            "file": fname,
            "title": "Age vs glucose scatter",
            "attribute": "quality",
            "type": "quality",
            "interpretation": (
                "Joint view of age and average glucose level colored by stroke outcome. "
                "Clusters or separation (if any) are exploratory only."
            ),
        }
    )

    return charts


def analyze_all_attributes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process every original dataset column automatically.
    For each column, select suitable analysis based on dtype/role.
    """
    charts: List[Dict[str, Any]] = []
    charts.extend(analyze_id_overview(df))
    charts.extend(analyze_numerical(df))
    charts.extend(analyze_categorical(df))
    charts.extend(analyze_target(df))

    cleaned, _ = clean_dataframe(df)
    charts.extend(analyze_quality(df, cleaned))

    # Catch-all: any unexpected extra columns still get a basic chart
    known = set(
        ["id"]
        + NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + BINARY_FEATURES
        + [TARGET_COLUMN]
    )
    for col in df.columns:
        if col in known:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df[col].dropna(), bins=25, color=TEAL, edgecolor="#0f172a")
            ax.set_title(f"Extra Numeric Attribute — {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            fname = _save(fig, f"extra_hist_{col}.png")
            charts.append(
                {
                    "file": fname,
                    "title": f"Extra attribute {col}",
                    "attribute": col,
                    "type": "extra",
                    "interpretation": (
                        f"Unexpected numeric column '{col}' was found and plotted for completeness."
                    ),
                }
            )
        else:
            counts = df[col].fillna("Missing").astype(str).value_counts().head(15)
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax, color=CYAN, edgecolor="#0f172a")
            ax.set_title(f"Extra Categorical Attribute — {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fname = _save(fig, f"extra_count_{col}.png")
            charts.append(
                {
                    "file": fname,
                    "title": f"Extra attribute {col}",
                    "attribute": col,
                    "type": "extra",
                    "interpretation": (
                        f"Unexpected categorical column '{col}' was found and plotted for completeness."
                    ),
                }
            )

    return charts


def run_eda() -> Dict[str, Any]:
    """Generate all EDA artifacts and write a catalog JSON for the website."""
    ensure_directories()
    df = load_dataset()
    overview = get_dataset_overview(df)
    # Ensure preprocessing summary exists for the preprocessing page
    try:
        prepare_splits(df)
    except Exception as exc:
        print(f"Warning: prepare_splits during EDA: {exc}")

    charts = analyze_all_attributes(df)
    # Drop entries without files from gallery listing (keep messages in JSON)
    catalog = {
        "n_charts": sum(1 for c in charts if c.get("file")),
        "charts": charts,
        "attributes_covered": sorted(
            {c["attribute"] for c in charts if c.get("attribute")}
        ),
        "dataset_shape": overview["shape"],
        "generated_from": str(BASE_DIR / "data" / "healthcare-dataset-stroke-data.csv"),
    }

    catalog_path = OUTPUTS_DIR / "eda_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Generated {catalog['n_charts']} charts into {GRAPHS_DIR}")
    print(f"Catalog saved to {catalog_path}")
    return catalog


if __name__ == "__main__":
    print("=" * 60)
    print("Brain StrokeSense — Exploratory Data Analysis")
    print("=" * 60)
    run_eda()
    print("EDA complete.")
