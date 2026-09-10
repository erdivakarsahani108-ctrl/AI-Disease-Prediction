"""Exploratory Data Analysis (EDA) & Data-Science dashboard chart builders.

All figures are built with Plotly so they are interactive in the Streamlit
app (hover, zoom, filter) as required by the project brief.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ml.data_loader import get_feature_columns


def disease_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["disease"].value_counts().reset_index()
    counts.columns = ["disease", "count"]
    fig = px.bar(
        counts, x="count", y="disease", orientation="h",
        title="Disease Distribution (record count per disease)",
        labels={"count": "Number of Records", "disease": "Disease"},
        color="count", color_continuous_scale="Teal",
    )
    fig.update_layout(height=900, yaxis={"categoryorder": "total ascending"})
    return fig


def category_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["disease_category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.pie(counts, names="category", values="count", title="Records by Disease Category (body system)")
    return fig


def symptom_frequency_chart(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    feature_cols = get_feature_columns(df)
    freq = df[feature_cols].sum().sort_values(ascending=False).head(top_n).reset_index()
    freq.columns = ["symptom", "count"]
    freq["symptom"] = freq["symptom"].str.replace("_", " ").str.title()
    fig = px.bar(
        freq, x="count", y="symptom", orientation="h",
        title=f"Top {top_n} Most Frequent Symptoms",
        labels={"count": "Occurrences", "symptom": "Symptom"},
        color="count", color_continuous_scale="Sunset",
    )
    fig.update_layout(height=700, yaxis={"categoryorder": "total ascending"})
    return fig


def disease_symptom_heatmap(df: pd.DataFrame, top_symptoms: int = 20, top_diseases: int = 20) -> go.Figure:
    feature_cols = get_feature_columns(df)
    top_syms = df[feature_cols].sum().sort_values(ascending=False).head(top_symptoms).index.tolist()
    top_diseases_list = df["disease"].value_counts().head(top_diseases).index.tolist()
    sub = df[df["disease"].isin(top_diseases_list)]
    matrix = sub.groupby("disease")[top_syms].mean()
    fig = px.imshow(
        matrix.values,
        x=[s.replace("_", " ").title() for s in top_syms],
        y=matrix.index.tolist(),
        color_continuous_scale="Viridis",
        title="Disease vs Symptom Heatmap (proportion of records with symptom)",
        labels=dict(color="Proportion"),
    )
    fig.update_layout(height=700)
    return fig


def class_imbalance_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["disease"].value_counts().reset_index()
    counts.columns = ["disease", "count"]
    fig = px.bar(
        counts.sort_values("count"), x="disease", y="count",
        title="Class Balance Across Diseases",
        labels={"count": "Number of Records", "disease": "Disease"},
    )
    fig.update_layout(xaxis_tickangle=-60, height=500)
    return fig


def real_vs_synthetic_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["synthetic_flag"].map({True: "Synthetic (augmented)", False: "Real (public dataset)"}).value_counts()
    fig = px.pie(names=counts.index, values=counts.values, title="Real vs Synthetic Records", hole=0.4)
    return fig


def symptom_count_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="symptom_count", nbins=20,
        title="Distribution of Number of Symptoms per Record",
        labels={"symptom_count": "Symptoms per Record"},
    )
    return fig
