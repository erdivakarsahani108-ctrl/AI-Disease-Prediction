"""Data Science Dashboard - EDA, model comparison, data quality, drift,
experiment tracking and feedback summary, all in one analytics page."""

import json

import pandas as pd
import streamlit as st

from src.database.feedback import feedback_summary
from src.ml.data_loader import load_dataset
from src.ml.experiment_tracker import load_experiments
from src.utils.paths import MODEL_COMPARISON_CSV, MODEL_METADATA_JSON, REPORTS_DIR
from src.utils.ui import render_sidebar_common
from src.visualization import eda_charts, model_charts

render_sidebar_common()
st.title("📊 Data Science Dashboard")
st.caption("Dataset overview, EDA, model comparison, data quality and monitoring - all computed from real pipeline artifacts.")

df = load_dataset()
metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8")) if MODEL_METADATA_JSON.exists() else {}

tab_overview, tab_eda, tab_models, tab_quality, tab_ops = st.tabs(
    ["📈 Overview", "🔬 EDA", "🤖 Model Comparison", "✅ Data Quality", "⚙️ Experiments & Drift"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Diseases", df["disease"].nunique())
    c3.metric("Symptom Features", 131)
    c4.metric("Best Model", metadata.get("best_model_name", "N/A"))
    m = metadata.get("metrics", {})
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
    c6.metric("Macro F1", f"{m.get('macro_f1', 0)*100:.1f}%")
    c7.metric("Weighted F1", f"{m.get('weighted_f1', 0)*100:.1f}%")
    c8.metric("CV F1 (macro)", f"{m.get('cv_mean_f1_macro', 0)*100:.1f}%")

    fb = feedback_summary()
    st.markdown("#### User Feedback Summary")
    if fb["total"]:
        fc1, fc2 = st.columns(2)
        fc1.metric("Total Feedback Received", fb["total"])
        fc2.metric("Marked Helpful", f"{fb['helpful_pct']}%")
    else:
        st.caption("No feedback recorded yet in this session/deployment.")

with tab_eda:
    st.plotly_chart(eda_charts.disease_distribution_chart(df), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(eda_charts.category_distribution_chart(df), use_container_width=True)
    c2.plotly_chart(eda_charts.real_vs_synthetic_chart(df), use_container_width=True)
    st.plotly_chart(eda_charts.symptom_frequency_chart(df), use_container_width=True)
    st.plotly_chart(eda_charts.disease_symptom_heatmap(df), use_container_width=True)
    st.plotly_chart(eda_charts.class_imbalance_chart(df), use_container_width=True)
    st.plotly_chart(eda_charts.symptom_count_distribution(df), use_container_width=True)

with tab_models:
    if MODEL_COMPARISON_CSV.exists():
        comp = pd.read_csv(MODEL_COMPARISON_CSV)
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.plotly_chart(model_charts.model_comparison_bar_chart(comp), use_container_width=True)
        st.plotly_chart(model_charts.model_timing_chart(comp), use_container_width=True)
        st.caption(
            "The best model is NOT chosen by accuracy alone - a weighted composite "
            "of macro-F1, weighted-F1, macro-recall and cross-validated F1-macro is "
            "used (see `scripts/train_model.py::SELECTION_WEIGHTS`)."
        )
        cm_path = REPORTS_DIR / "confusion_matrix.html"
        if cm_path.exists():
            st.markdown("#### Confusion Matrix")
            st.components.v1.html(cm_path.read_text(encoding="utf-8"), height=1000, scrolling=True)
    else:
        st.info("Run `python scripts/train_model.py` to generate model comparison data.")

with tab_quality:
    quality_path = REPORTS_DIR / "data_quality_report.json"
    if quality_path.exists():
        report = json.loads(quality_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Data Quality Score", f"{report['data_quality_score']}/100")
        c2.metric("Real Records", f"{report['real_pct']}%")
        c3.metric("Class Imbalance Ratio", f"{report['class_imbalance_ratio']}:1")
        st.json(report)
        st.caption(
            "Score = 100 minus penalties for missing values, duplicates, invalid "
            "records, class imbalance and synthetic-data proportion. See "
            "docs/data_quality_report.md for the full methodology."
        )
    else:
        st.info("Run `python scripts/validate_dataset.py` to generate the data quality report.")

with tab_ops:
    st.markdown("#### Experiment Tracking Log")
    exp_df = load_experiments()
    if not exp_df.empty:
        st.dataframe(exp_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No experiments logged yet.")

    st.markdown("#### Model / Data Drift Monitor")
    st.info(
        "Drift detection compares recently logged prediction-input symptom "
        "distributions against the training baseline (`models/drift_baseline.json`). "
        "Not enough live prediction traffic has been logged yet in this session to "
        "compute a meaningful drift estimate - see `src/ml/drift.py`."
    )
