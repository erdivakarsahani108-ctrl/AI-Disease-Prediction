"""Interactive Disease-Symptom-Category Knowledge Graph.

Nodes: Disease, Symptom, Category
Edges: Disease -> HAS_SYMPTOM -> Symptom ; Disease -> BELONGS_TO -> Category

Built with networkx (graph structure) + Plotly (interactive rendering),
so users can filter/explore by disease, symptom or category directly in
the Streamlit app.
"""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go

from src.preprocessing.categories import get_category
from src.preprocessing.clean import build_disease_symptom_pool, load_real_disease_symptom_sets
from src.utils.text import to_display_name


def build_graph(max_diseases: int | None = None, focus_disease: str | None = None) -> nx.Graph:
    real_df = load_real_disease_symptom_sets()
    pool = build_disease_symptom_pool(real_df)

    diseases = sorted(pool.keys())
    if focus_disease:
        diseases = [focus_disease] if focus_disease in pool else []
    elif max_diseases:
        diseases = diseases[:max_diseases]

    G = nx.Graph()
    for disease in diseases:
        category = get_category(disease)
        G.add_node(disease, type="disease", label=disease)
        G.add_node(f"CAT::{category}", type="category", label=category)
        G.add_edge(disease, f"CAT::{category}", relation="BELONGS_TO")
        for symptom in pool.get(disease, []):
            symptom_node = f"SYM::{symptom}"
            G.add_node(symptom_node, type="symptom", label=to_display_name(symptom))
            G.add_edge(disease, symptom_node, relation="HAS_SYMPTOM")
    return G


NODE_COLORS = {"disease": "#e74c3c", "symptom": "#3498db", "category": "#2ecc71"}
NODE_SIZES = {"disease": 22, "symptom": 10, "category": 26}


def render_graph(G: nx.Graph) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure().update_layout(title="No data to display for this filter.")

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color="#bbbbbb"), hoverinfo="none", mode="lines")

    traces = [edge_trace]
    for node_type in ("category", "disease", "symptom"):
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        if not nodes:
            continue
        traces.append(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers+text" if node_type != "symptom" else "markers",
                text=[G.nodes[n]["label"] for n in nodes],
                textposition="top center",
                hovertext=[f"{node_type.title()}: {G.nodes[n]['label']}" for n in nodes],
                hoverinfo="text",
                marker=dict(size=NODE_SIZES[node_type], color=NODE_COLORS[node_type], line=dict(width=1, color="white")),
                name=node_type.title(),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Disease - Symptom - Category Knowledge Graph",
        showlegend=True,
        hovermode="closest",
        height=750,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
