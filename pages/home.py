"""Home page - project overview, key stats, quick navigation guidance."""

import json

import pandas as pd
import streamlit as st

from src.ml.data_loader import load_dataset
from src.utils.paths import DISEASE_DICTIONARY_CSV, MEDICINE_DICTIONARY_CSV, MODEL_METADATA_JSON
from src.utils.ui import render_disclaimer_banner, render_sidebar_common

render_sidebar_common()

st.title("🩺 AI-Based Disease Prediction & Intelligent Health Assistant")
st.caption("Data Science + Machine Learning + NLP + Explainable AI + Knowledge Graph - Final Year B.Tech Project")

render_disclaimer_banner()

try:
    df = load_dataset()
    metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8")) if MODEL_METADATA_JSON.exists() else {}
except FileNotFoundError:
    df = None
    metadata = {}

col1, col2, col3, col4 = st.columns(4)
if df is not None:
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Diseases Covered", df["disease"].nunique())
    col3.metric("Symptom Features", len([c for c in df.columns if c not in {
        'record_id','disease','disease_category','symptoms','symptom_count','source','synthetic_flag'
    }]))
    col4.metric("Best Model Accuracy", f"{metadata.get('metrics', {}).get('accuracy', 0)*100:.1f}%" if metadata else "N/A")
else:
    st.warning("Dataset not found. Run `python scripts/prepare_data.py` to build it.")

st.markdown("---")

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        """
        <div class="app-card">
        <h4>What this system does</h4>
        <ul>
            <li>Understands natural-language symptom descriptions in <b>English, Hindi and Hinglish</b></li>
            <li>Extracts and normalizes multiple symptoms (with negation handling)</li>
            <li>Predicts <b>Top-1 / Top-3 / Top-5</b> possible diseases with model confidence</li>
            <li>Explains <i>why</i> using Explainable AI (feature contribution)</li>
            <li>Detects potential <b>emergency / red-flag</b> symptoms first, before any prediction</li>
            <li>Provides a structured Disease Dictionary, Medicine Information and Knowledge Graph</li>
            <li>Offers 2D/3D Data-Science visualizations and a downloadable Health Report</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="app-card">
        <h4>How to use this app</h4>
        <ol>
            <li>Go to <b>🤖 AI Health Assistant</b> to chat about your symptoms conversationally, or</li>
            <li>Go to <b>🩺 Disease Prediction</b> to pick symptoms from a structured list</li>
            <li>Review the Top predictions, explanation and risk indicator</li>
            <li>Explore <b>📖 Disease Dictionary</b> / <b>💊 Medicine Information</b> for general knowledge</li>
            <li>Download a <b>📄 Health Report</b> summarizing your session</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    if metadata:
        st.markdown(
            f"""
            <div class="app-card">
            <h4>Model Snapshot</h4>
            <p><b>Best model:</b> {metadata.get('best_model_name', 'N/A')}</p>
            <p><b>Accuracy:</b> {metadata.get('metrics', {}).get('accuracy', 0)*100:.1f}%</p>
            <p><b>Macro F1:</b> {metadata.get('metrics', {}).get('macro_f1', 0)*100:.1f}%</p>
            <p><b>Cross-validated F1:</b> {metadata.get('metrics', {}).get('cv_mean_f1_macro', 0)*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Train the model first: `python scripts/train_model.py`")

    st.markdown(
        """
        <div class="app-card">
        <h4>This is NOT</h4>
        <ul>
            <li>A replacement for professional medical diagnosis</li>
            <li>An automatic medicine-prescription system</li>
            <li>A clinical-grade risk-scoring tool</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
