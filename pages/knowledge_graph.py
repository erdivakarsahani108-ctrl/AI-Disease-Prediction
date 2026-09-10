"""Disease-Symptom-Category Knowledge Graph - interactive exploration."""

import streamlit as st

from src.preprocessing.clean import build_disease_symptom_pool, load_real_disease_symptom_sets
from src.utils.ui import render_sidebar_common
from src.visualization import knowledge_graph as kg

render_sidebar_common()
st.title("🔗 Disease-Symptom Knowledge Graph")
st.caption("Nodes: Disease / Symptom / Category. Edge: Disease -HAS_SYMPTOM-> Symptom. Click filters below to explore.")

pool = build_disease_symptom_pool(load_real_disease_symptom_sets())
all_diseases = sorted(pool.keys())

mode = st.radio("Explore mode", ["All diseases (limited)", "Focus on one disease"], horizontal=True)

if mode == "Focus on one disease":
    focus = st.selectbox("Select a disease", all_diseases)
    graph = kg.build_graph(focus_disease=focus)
else:
    max_n = st.slider("Number of diseases to show (larger graphs are slower to render)", 3, 20, 8)
    graph = kg.build_graph(max_diseases=max_n)

st.plotly_chart(kg.render_graph(graph), use_container_width=True)
st.caption(f"Graph currently shows {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
