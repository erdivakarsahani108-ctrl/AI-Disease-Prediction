"""Shared Streamlit UI helpers - modern styling, disclaimer banners, sidebar
components - reused across every page so the app looks and behaves
consistently (per the "Modern Frontend" requirement).
"""

from __future__ import annotations

import streamlit as st

from src.utils.session import clear_session

CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .disclaimer-banner {
        background: linear-gradient(90deg, #fff3cd, #ffeeba);
        border-left: 5px solid #e67e22;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #6b4b17;
    }
    .app-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eef0f3;
        margin-bottom: 0.9rem;
    }
    .emergency-banner {
        background: #fdecea;
        border-left: 6px solid #e74c3c;
        padding: 1rem;
        border-radius: 8px;
        color: #7a1f13;
        font-weight: 500;
    }
    .risk-low { color: #1e8e3e; font-weight: 700; }
    .risk-moderate { color: #b8860b; font-weight: 700; }
    .risk-high { color: #d93025; font-weight: 700; }
    footer {visibility: hidden;}
</style>
"""


def apply_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_disclaimer_banner():
    st.markdown(
        """
        <div class="disclaimer-banner">
        ⚠️ <b>Medical Disclaimer:</b> This is an AI-assisted educational and
        informational tool - <b>NOT a medical diagnosis system</b>. Predictions
        are statistical model outputs, not medically validated probabilities.
        Always consult a qualified healthcare professional for diagnosis and
        treatment. In an emergency, contact your local emergency services
        immediately.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_common():
    with st.sidebar:
        st.markdown("### 🩺 AI Disease Prediction")
        st.caption("AI-Based Disease Prediction & Intelligent Health Assistant")
        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            clear_session()
            st.rerun()
        st.caption(
            "Clearing the session removes all symptoms and chat history from "
            "memory. No personal identifiers are collected or stored by this app."
        )
        st.divider()
        st.caption("Not a substitute for professional medical advice.")
