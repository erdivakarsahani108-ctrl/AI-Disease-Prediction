"""About page - project information, policy links, and disclaimers."""

import streamlit as st

from src.utils.paths import DOCS_DIR
from src.utils.ui import render_sidebar_common

render_sidebar_common()
st.title("ℹ️ About This Project")

st.markdown(
    """
### AI-Based Disease Prediction & Intelligent Health Assistant

A Data Science + AI/ML + NLP + Explainable AI Final Year B.Tech project that
combines a real public disease-symptom dataset, a documented synthetic-
augmentation methodology, classical ML model comparison, a hybrid rule-based
+ ML + knowledge-base prediction pipeline, and a modern conversational
Streamlit frontend.

**This system is an educational, AI-assisted information tool. It is NOT a
medical diagnosis system and does not replace professional healthcare.**
"""
)

st.markdown("### Project Documentation")
doc_links = {
    "Medical Disclaimer": "medical_disclaimer.md",
    "Privacy Policy": "privacy_policy.md",
    "Data Sources": "data_sources.md",
    "Dataset Methodology": "dataset_methodology.md",
    "Data Quality Report": "data_quality_report.md",
    "Model Card": "model_card.md",
    "AI Transparency": "ai_transparency.md",
    "Responsible AI": "responsible_ai.md",
    "Security": "security.md",
    "Architecture": "architecture.md",
    "Limitations": "limitations.md",
    "Model Evaluation": "model_evaluation.md",
    "Data Dictionary": "data_dictionary.md",
}

cols = st.columns(3)
for i, (label, filename) in enumerate(doc_links.items()):
    path = DOCS_DIR / filename
    with cols[i % 3]:
        with st.expander(label):
            if path.exists():
                st.markdown(path.read_text(encoding="utf-8"))
            else:
                st.caption("Document not yet generated.")

st.markdown("### Contact / Feedback")
st.info(
    "This is an academic project. For questions, issues, or feedback about "
    "this repository, please use the project's issue tracker / repository "
    "contact channel (see `docs/contact_feedback.md`). No personal data is "
    "collected by this contact process within the app itself."
)

st.markdown("### License")
st.caption("See `LICENSE` in the project root.")
