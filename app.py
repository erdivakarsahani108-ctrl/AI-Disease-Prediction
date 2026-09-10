"""
app.py
=======
Streamlit application entry point. Sets global page config, applies the
shared modern theme, initializes session state and wires up the
multi-page navigation (Home, AI Health Assistant, Disease Prediction,
Disease Dictionary, Medicine Information, Data Science Dashboard,
2D/3D Visualization, Knowledge Graph, Health Report, About).

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.utils.session import init_session_state
from src.utils.ui import apply_theme

st.set_page_config(
    page_title="AI Disease Prediction & Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
init_session_state()

home = st.Page("pages/home.py", title="Home", icon="🏠", default=True)
chat = st.Page("pages/chat_assistant.py", title="AI Health Assistant", icon="🤖")
predict = st.Page("pages/prediction.py", title="Disease Prediction", icon="🩺")
dictionary = st.Page("pages/disease_dictionary.py", title="Disease Dictionary", icon="📖")
medicine = st.Page("pages/medicine_information.py", title="Medicine Information", icon="💊")
analytics = st.Page("pages/analytics.py", title="Data Science Dashboard", icon="📊")
viz = st.Page("pages/visualization.py", title="2D / 3D Visualization", icon="🧬")
kg = st.Page("pages/knowledge_graph.py", title="Knowledge Graph", icon="🔗")
report = st.Page("pages/health_report.py", title="Health Report", icon="📄")
about = st.Page("pages/about.py", title="About", icon="ℹ️")

pg = st.navigation(
    {
        "Overview": [home],
        "AI Assistant": [chat, predict],
        "Knowledge": [dictionary, medicine, kg],
        "Data Science": [analytics, viz],
        "Reports & Info": [report, about],
    }
)
pg.run()
