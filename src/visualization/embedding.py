"""2D/3D embedding visualizations (PCA / t-SNE) of the mathematical
symptom feature space.

IMPORTANT (per project brief): these are visualizations of an abstract,
high-dimensional SYMPTOM feature space projected down to 2D/3D using
standard dimensionality-reduction techniques. They are a Data-Science
visualization of the dataset's structure - they must NOT be interpreted as
real human anatomy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.ml.data_loader import get_feature_columns
from src.utils.paths import RANDOM_SEED

ANATOMY_DISCLAIMER = (
    "Note: this is a mathematical projection of the symptom feature space "
    "(PCA/t-SNE), not a depiction of real human anatomy."
)


def _sample_for_speed(df: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=RANDOM_SEED)


def pca_2d_scatter(df: pd.DataFrame, color_by: str = "disease_category") -> go.Figure:
    sample = _sample_for_speed(df)
    feature_cols = get_feature_columns(sample)
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(sample[feature_cols])
    plot_df = sample[["disease", "disease_category", "synthetic_flag"]].copy()
    plot_df["PC1"], plot_df["PC2"] = coords[:, 0], coords[:, 1]
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color=color_by, hover_data=["disease"],
        title=f"2D PCA Embedding of Symptom Feature Space (colored by {color_by})",
    )
    fig.add_annotation(text=ANATOMY_DISCLAIMER, xref="paper", yref="paper", x=0, y=-0.12, showarrow=False, font=dict(size=10, color="gray"))
    return fig


def pca_3d_scatter(df: pd.DataFrame, color_by: str = "disease_category") -> go.Figure:
    sample = _sample_for_speed(df)
    feature_cols = get_feature_columns(sample)
    coords = PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(sample[feature_cols])
    plot_df = sample[["disease", "disease_category", "synthetic_flag"]].copy()
    plot_df["PC1"], plot_df["PC2"], plot_df["PC3"] = coords[:, 0], coords[:, 1], coords[:, 2]
    fig = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3", color=color_by, hover_data=["disease"],
        title=f"3D PCA Feature-Space Visualization (colored by {color_by})",
    )
    fig.update_layout(
        annotations=[dict(text=ANATOMY_DISCLAIMER, xref="paper", yref="paper", x=0, y=-0.05, showarrow=False, font=dict(size=10, color="gray"))]
    )
    return fig


def tsne_2d_scatter(df: pd.DataFrame, color_by: str = "disease_category", perplexity: int = 30) -> go.Figure:
    sample = _sample_for_speed(df, max_rows=1500)
    feature_cols = get_feature_columns(sample)
    perplexity = min(perplexity, max(5, len(sample) // 10))
    coords = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity, init="pca").fit_transform(
        sample[feature_cols]
    )
    plot_df = sample[["disease", "disease_category"]].copy()
    plot_df["Dim1"], plot_df["Dim2"] = coords[:, 0], coords[:, 1]
    fig = px.scatter(
        plot_df, x="Dim1", y="Dim2", color=color_by, hover_data=["disease"],
        title=f"t-SNE 2D Embedding Map (colored by {color_by})",
    )
    fig.add_annotation(text=ANATOMY_DISCLAIMER, xref="paper", yref="paper", x=0, y=-0.12, showarrow=False, font=dict(size=10, color="gray"))
    return fig


def disease_cluster_centroids_3d(df: pd.DataFrame) -> go.Figure:
    """3D PCA scatter of DISEASE-level centroids (mean symptom vector per
    disease) - useful for seeing which diseases are 'close' in symptom
    space (Disease Clustering)."""
    feature_cols = get_feature_columns(df)
    centroids = df.groupby("disease")[feature_cols].mean()
    coords = PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(centroids)
    plot_df = pd.DataFrame(
        {"disease": centroids.index, "PC1": coords[:, 0], "PC2": coords[:, 1], "PC3": coords[:, 2]}
    )
    category_map = df.drop_duplicates("disease").set_index("disease")["disease_category"]
    plot_df["category"] = plot_df["disease"].map(category_map)
    fig = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3", color="category", text="disease",
        title="3D Disease Clustering (centroid of symptom feature space per disease)",
    )
    fig.update_traces(textposition="top center")
    return fig
