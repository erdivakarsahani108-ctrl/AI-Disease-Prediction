"""Health Report - downloadable summary of the current session.

Explicitly NOT a medical diagnosis certificate.
"""

import streamlit as st

from src.utils.report_generator import generate_health_report
from src.utils.session import init_session_state
from src.utils.ui import render_disclaimer_banner, render_sidebar_common

init_session_state()
render_sidebar_common()
st.title("📄 Health Report")
render_disclaimer_banner()

result = st.session_state.get("last_result")

if result is None or result.status != "ok":
    st.info(
        "No completed prediction is available yet in this session. Go to "
        "**🤖 AI Health Assistant** or **🩺 Disease Prediction**, get a "
        "prediction, then return here to download your report."
    )
else:
    st.markdown("### Report Preview")
    st.write("**Symptoms:**", ", ".join(s.replace("_", " ").title() for s in result.active_symptoms))
    if result.extraction and result.extraction.duration_days:
        st.write("**Duration:**", f"{result.extraction.duration_days} day(s)")
    st.write("**Top predictions:**")
    for p in result.top_predictions:
        st.write(f"- {p['disease']} ({p['confidence']*100:.1f}%)")
    if result.risk:
        st.write("**Risk indicator:**", f"{result.risk['emoji']} {result.risk['band']}")

    pdf_bytes = generate_health_report(
        active_symptoms=result.active_symptoms,
        duration_days=result.extraction.duration_days if result.extraction else None,
        top_predictions=result.top_predictions,
        risk=result.risk,
        disease_info=result.disease_info,
    )
    st.download_button(
        "⬇️ Download Health Report (PDF)",
        data=pdf_bytes,
        file_name="ai_health_report.pdf",
        mime="application/pdf",
        type="primary",
    )
    st.caption(
        "This report is NOT a medical diagnosis certificate. It summarizes "
        "AI-assisted, model-based information generated during this session only."
    )
