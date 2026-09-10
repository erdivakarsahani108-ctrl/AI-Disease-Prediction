"""2D/3D Visualization - embedding maps, disease clustering, and a
schematic body-region diagram."""

import streamlit as st

from src.ml.data_loader import load_dataset
from src.utils.session import init_session_state
from src.utils.ui import render_sidebar_common
from src.visualization import body_diagram, embedding

init_session_state()
render_sidebar_common()
st.title("🧬 2D / 3D Data-Science Visualization")
st.caption(
    "These charts visualize the mathematical SYMPTOM FEATURE SPACE of the dataset "
    "(via PCA/t-SNE dimensionality reduction) - they are NOT depictions of real "
    "human anatomy."
)

df = load_dataset()

tab1, tab2, tab3, tab4 = st.tabs(["2D Embedding Map", "3D Feature Space", "3D Disease Clustering", "Body-Region Summary"])

with tab1:
    color_by = st.selectbox("Color by", ["disease_category", "disease", "synthetic_flag"], key="c1")
    method = st.radio("Method", ["PCA", "t-SNE"], horizontal=True)
    if method == "PCA":
        st.plotly_chart(embedding.pca_2d_scatter(df, color_by=color_by), use_container_width=True)
    else:
        with st.spinner("Computing t-SNE embedding (this can take a moment)..."):
            st.plotly_chart(embedding.tsne_2d_scatter(df, color_by=color_by), use_container_width=True)

with tab2:
    color_by2 = st.selectbox("Color by", ["disease_category", "disease", "synthetic_flag"], key="c2")
    st.plotly_chart(embedding.pca_3d_scatter(df, color_by=color_by2), use_container_width=True)

with tab3:
    st.plotly_chart(embedding.disease_cluster_centroids_3d(df), use_container_width=True)
    st.caption("Each point is the mean symptom-vector (centroid) of one disease, projected to 3D via PCA.")

with tab4:
    st.caption("Illustrative schematic only - built from simple shapes, not a medical illustration.")
    active = st.session_state.get("known_symptoms", set())
    if active:
        st.info(f"Using {len(active)} symptom(s) from your current session (Chat / Prediction page).")
    st.plotly_chart(body_diagram.body_region_figure(active), use_container_width=True)
