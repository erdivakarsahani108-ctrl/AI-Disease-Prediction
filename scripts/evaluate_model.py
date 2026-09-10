"""
scripts/evaluate_model.py
===========================
Reproducible Data-Science pipeline step 3: MODEL EVALUATION.

Loads the persisted best model + encoders, re-creates the exact same
stratified train/test split (same random seed), and produces:

    reports/confusion_matrix.html      (interactive Plotly heatmap)
    reports/classification_report.csv  (per-class precision/recall/F1)
    docs/model_evaluation.md           (human-readable summary)

Run:
    python scripts/evaluate_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

from src.ml.data_loader import load_dataset, train_test_split_dataset
from src.utils.paths import (
    BEST_MODEL_PKL,
    DOCS_DIR,
    ENCODERS_PKL,
    MODEL_METADATA_JSON,
    REPORTS_DIR,
)


def main():
    if not BEST_MODEL_PKL.exists():
        raise FileNotFoundError("No trained model found. Run `python scripts/train_model.py` first.")

    model = joblib.load(BEST_MODEL_PKL)
    encoders = joblib.load(ENCODERS_PKL)
    label_encoder = encoders["label_encoder"]
    metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8"))

    df = load_dataset()
    X_train, X_test, y_train, y_test, _, feature_cols = train_test_split_dataset(df)

    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORTS_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=f"Confusion Matrix - {metadata['best_model_name']}",
        color_continuous_scale="Blues",
    )
    fig.update_layout(width=1000, height=1000)
    fig.write_html(REPORTS_DIR / "confusion_matrix.html")

    lines = [
        "# Model Evaluation Report\n",
        f"**Best model:** {metadata['best_model_name']}\n",
        f"**Dataset version hash:** {metadata['dataset_version_hash']}\n",
        f"**Train size:** {metadata['train_size']}  |  **Test size:** {metadata['test_size']}\n",
        "## Overall Metrics (held-out test set)\n",
        f"- Accuracy: **{metadata['metrics']['accuracy']:.4f}**",
        f"- Macro Precision: **{metadata['metrics']['macro_precision']:.4f}**",
        f"- Macro Recall: **{metadata['metrics']['macro_recall']:.4f}**",
        f"- Macro F1: **{metadata['metrics']['macro_f1']:.4f}**",
        f"- Weighted F1: **{metadata['metrics']['weighted_f1']:.4f}**",
        f"- Cross-validation mean F1-macro ({metadata['metrics']['cv_folds']}-fold): **{metadata['metrics']['cv_mean_f1_macro']:.4f}** (+/- {metadata['metrics']['cv_std_f1_macro']:.4f})",
        "\n## Per-class report\n",
        "See `reports/classification_report.csv` for full per-disease precision/recall/F1/support.",
        "\n## Confusion Matrix\n",
        "See `reports/confusion_matrix.html` (interactive Plotly heatmap).",
        "\n## Model Comparison\n",
        "See `models/model_comparison.csv` for all 6 candidate models evaluated.",
        "\n## Notes on Metric Choice\n",
        (
            "Because several diseases are represented by far fewer real records "
            "than others (see docs/data_quality_report.md), **macro-averaged** "
            "metrics (which weight every class equally) are used as the primary "
            "model-selection criteria rather than accuracy or weighted metrics "
            "alone, so that rare diseases are not ignored by the selection process."
        ),
    ]
    (DOCS_DIR / "model_evaluation.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print(f" - {REPORTS_DIR / 'classification_report.csv'}")
    print(f" - {REPORTS_DIR / 'confusion_matrix.html'}")
    print(f" - {DOCS_DIR / 'model_evaluation.md'}")
    print("\nOverall metrics:", metadata["metrics"])


if __name__ == "__main__":
    main()
