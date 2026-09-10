"""AI Health Assistant - ChatGPT-style conversational symptom interview."""

import streamlit as st

from src.database.feedback import save_feedback
from src.prediction.pipeline import run_pipeline
from src.utils.session import init_session_state
from src.utils.ui import render_disclaimer_banner, render_sidebar_common

init_session_state()
render_sidebar_common()

st.title("🤖 AI Health Assistant")
st.caption("Describe your symptoms in your own words - English, Hindi or Hinglish all work.")
render_disclaimer_banner()

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.known_symptoms = set()
        st.session_state.denied_symptoms = set()
        st.session_state.last_result = None
        st.rerun()

if not st.session_state.chat_history:
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your AI Health Assistant. Please describe your symptoms "
                "(e.g. *\"Mujhe 3 din se bukhar aur sir dard hai\"* or *\"I have fever, "
                "cough and body pain\"*)."
            ),
        }
    )

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Describe your symptoms...")

if user_text:
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your symptoms..."):
            result = run_pipeline(
                user_text,
                known_symptoms=st.session_state.known_symptoms,
                denied_symptoms=st.session_state.denied_symptoms,
            )
        st.session_state.known_symptoms = set(result.active_symptoms)
        st.session_state.last_result = result

        if result.status == "emergency":
            st.markdown(f'<div class="emergency-banner">{result.message}</div>', unsafe_allow_html=True)
            for r in result.emergency_reasons:
                st.markdown(f"- {r}")
            reply = "Emergency indicators detected - please seek immediate professional care."

        elif result.status == "insufficient_info":
            st.markdown(result.message)
            if result.active_symptoms:
                st.caption("Symptoms understood so far: " + ", ".join(s.replace("_", " ").title() for s in result.active_symptoms))
            st.progress(result.completeness_pct / 100, text=f"Information completeness: {result.completeness_pct}%")
            if result.next_question:
                st.markdown(f"**{result.next_question['question']}**")
            reply = result.message

        else:
            st.markdown("Here's what I found based on your symptoms:")
            st.caption("Active symptoms: " + ", ".join(s.replace("_", " ").title() for s in result.active_symptoms))
            st.progress(result.completeness_pct / 100, text=f"Information completeness: {result.completeness_pct}%")

            st.markdown("#### Possible Conditions")
            for i, p in enumerate(result.top_predictions, start=1):
                st.markdown(f"**{i}. {p['disease']}** — {p['confidence']*100:.1f}% model confidence")
                st.progress(min(1.0, p["confidence"]))

            if result.risk:
                risk_class = {"Low Risk": "risk-low", "Moderate Risk": "risk-moderate", "High Risk": "risk-high"}[result.risk["band"]]
                st.markdown(f'{result.risk["emoji"]} <span class="{risk_class}">{result.risk["band"]}</span>', unsafe_allow_html=True)
                st.caption(result.risk["disclaimer"])

            with st.expander("🔍 Why this prediction? (Explainable AI)"):
                if result.explanation:
                    st.caption(f"Method: {result.explanation['method']}")
                    for symptom, score in result.explanation["contributions"]:
                        st.write(f"✓ {symptom.replace('_', ' ').title()} (contribution: {score:.3f})")
                if result.comparison_table is not None:
                    st.markdown("**Comparison across Top predictions:**")
                    st.dataframe(result.comparison_table, use_container_width=True, hide_index=True)

            if result.next_question:
                st.markdown(f"**Follow-up:** {result.next_question['question']}")

            st.caption("⚠️ Model confidence is not a medically validated probability. This is model-based prediction, not diagnosis.")

            if result.top_predictions:
                fb_col1, fb_col2 = st.columns(2)
                if fb_col1.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_history)}"):
                    save_feedback(result.top_predictions[0]["disease"], result.top_predictions[0]["confidence"], True, symptom_count=len(result.active_symptoms))
                    st.toast("Thanks for your feedback!")
                if fb_col2.button("👎 Not Helpful", key=f"nothelpful_{len(st.session_state.chat_history)}"):
                    save_feedback(result.top_predictions[0]["disease"], result.top_predictions[0]["confidence"], False, symptom_count=len(result.active_symptoms))
                    st.toast("Thanks for your feedback!")

            reply = f"Top prediction: {result.top_predictions[0]['disease']} ({result.top_predictions[0]['confidence']*100:.1f}% confidence)."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
