"""Disease Prediction - structured symptom-picker (multiselect) alternative
to the free-text chat assistant, feeding the SAME hybrid prediction
pipeline (NLP -> Safety -> ML -> Explainability -> Risk)."""

import pandas as pd
import streamlit as st

from src.database.feedback import save_feedback
from src.prediction.pipeline import run_pipeline
from src.utils.paths import SYMPTOM_DICTIONARY_CSV
from src.utils.session import init_session_state
from src.utils.text import to_display_name
from src.utils.ui import render_disclaimer_banner, render_sidebar_common

init_session_state()
render_sidebar_common()

st.title("🩺 Disease Prediction (Structured Symptom Picker)")
st.caption("Pick your symptoms from the list below - useful if you prefer not to type free text.")
render_disclaimer_banner()

symptom_df = pd.read_csv(SYMPTOM_DICTIONARY_CSV)
symptom_options = {to_display_name(s): s for s in sorted(symptom_df["symptom"])}

selected_labels = st.multiselect(
    "Select all symptoms that apply:",
    options=list(symptom_options.keys()),
    help="Start typing to search. You can select multiple symptoms.",
)
duration = st.slider("How many days have you had these symptoms?", 0, 30, 0)
top_k = st.slider("How many possible conditions to show?", 1, 5, 5)

if st.button("🔍 Predict", type="primary", disabled=not selected_labels):
    display_text = ", ".join(selected_labels)
    if duration:
        display_text += f", since {duration} days"

    with st.spinner("Running hybrid AI prediction pipeline..."):
        result = run_pipeline(display_text, top_k=top_k)

    if result.status == "emergency":
        st.markdown(f'<div class="emergency-banner">{result.message}</div>', unsafe_allow_html=True)
        for r in result.emergency_reasons:
            st.markdown(f"- {r}")

    elif result.status == "insufficient_info":
        st.warning(result.message)
        st.progress(result.completeness_pct / 100, text=f"Information completeness: {result.completeness_pct}%")

    else:
        st.progress(result.completeness_pct / 100, text=f"Information completeness: {result.completeness_pct}%")
        st.markdown("### Possible Conditions")
        cols = st.columns(len(result.top_predictions))
        for col, p in zip(cols, result.top_predictions):
            with col:
                st.markdown(f'<div class="app-card"><b>{p["disease"]}</b><br>{p["confidence"]*100:.1f}% confidence</div>', unsafe_allow_html=True)

        if result.risk:
            risk_class = {"Low Risk": "risk-low", "Moderate Risk": "risk-moderate", "High Risk": "risk-high"}[result.risk["band"]]
            st.markdown(f'{result.risk["emoji"]} <span class="{risk_class}">{result.risk["band"]}</span> (severity score: {result.risk["score"]})', unsafe_allow_html=True)
            st.caption(result.risk["disclaimer"])

        with st.expander("🔍 Explainable AI - Why this prediction?", expanded=True):
            if result.explanation:
                st.caption(f"Method: {result.explanation['method']}")
                imp_df = pd.DataFrame(result.explanation["contributions"], columns=["Symptom", "Contribution"])
                imp_df["Symptom"] = imp_df["Symptom"].apply(to_display_name)
                st.bar_chart(imp_df.set_index("Symptom"))
            if result.comparison_table is not None:
                st.markdown("**Differential comparison across Top predictions:**")
                st.dataframe(result.comparison_table, use_container_width=True, hide_index=True)

        if result.disease_info:
            with st.expander(f"📖 About {result.top_predictions[0]['disease']}"):
                st.write(result.disease_info.get("description", ""))
                st.markdown(f"**General precautions:** {result.disease_info.get('general_precautions', '')}")
                st.markdown(f"**Similar diseases:** {result.disease_info.get('similar_diseases', '')}")
                st.caption(f"Source: {result.disease_info.get('source', '')} (verified {result.disease_info.get('verification_date', '')})")

        st.caption("⚠️ Model confidence is not a medically validated probability. This is model-based prediction, not diagnosis.")

        st.session_state.known_symptoms = set(result.active_symptoms)
        st.session_state.last_result = result

        fb1, fb2 = st.columns(2)
        if fb1.button("👍 Helpful"):
            save_feedback(result.top_predictions[0]["disease"], result.top_predictions[0]["confidence"], True, symptom_count=len(result.active_symptoms))
            st.toast("Thanks for your feedback!")
        if fb2.button("👎 Not Helpful"):
            save_feedback(result.top_predictions[0]["disease"], result.top_predictions[0]["confidence"], False, symptom_count=len(result.active_symptoms))
            st.toast("Thanks for your feedback!")
