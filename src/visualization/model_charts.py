"""Model comparison & feature-importance chart builders for the Data Science
Dashboard / Model Comparison page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def model_comparison_bar_chart(comparison_df: pd.DataFrame) -> go.Figure:
    metrics = ["accuracy", "macro_f1", "weighted_f1", "cv_mean_f1_macro"]
    melted = comparison_df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    fig = px.bar(
        melted, x="model", y="score", color="metric", barmode="group",
        title="Model Comparison - Accuracy / Macro-F1 / Weighted-F1 / CV F1-macro",
        labels={"score": "Score", "model": "Model"},
    )
    fig.update_layout(yaxis_range=[0, 1], height=500)
    return fig


def model_timing_chart(comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train time (s)", x=comparison_df["model"], y=comparison_df["train_time_sec"]))
    fig.add_trace(go.Bar(name="Predict time (s)", x=comparison_df["model"], y=comparison_df["predict_time_sec"]))
    fig.update_layout(title="Training / Prediction Time by Model", barmode="group", yaxis_type="log", height=450)
    return fig


def feature_importance_chart(feature_names: list[str], importances, top_n: int = 15, title: str = "Feature Importance") -> go.Figure:
    df = pd.DataFrame({"symptom": feature_names, "importance": importances})
    df["symptom"] = df["symptom"].str.replace("_", " ").str.title()
    df = df.reindex(df["importance"].abs().sort_values(ascending=False).index).head(top_n)
    fig = px.bar(
        df.sort_values("importance"), x="importance", y="symptom", orientation="h",
        title=title, labels={"importance": "Importance", "symptom": "Symptom"},
        color="importance", color_continuous_scale="RdBu",
    )
    fig.update_layout(height=550)
    return fig
