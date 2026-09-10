"""Simple, original, license-free 2D schematic body-region diagram.

This draws a stylized human silhouette using basic Plotly shapes (NOT a
copied medical illustration or third-party asset) and overlays how many of
the user's currently-active symptoms map to each broad body region, purely
as an illustrative summary aid.

Mapping is intentionally coarse and organizational (not a clinical claim).
"""

from __future__ import annotations

import plotly.graph_objects as go

BODY_REGION_SYMPTOM_MAP = {
    "Head / Neurological": [
        "headache", "dizziness", "loss_of_balance", "unsteadiness", "slurred_speech",
        "altered_sensorium", "coma", "spinning_movements", "lack_of_concentration",
        "visual_disturbances", "blurred_and_distorted_vision", "pain_behind_the_eyes",
        "stiff_neck", "neck_pain", "loss_of_smell", "headache",
    ],
    "Chest / Respiratory": [
        "cough", "breathlessness", "chest_pain", "phlegm", "mucoid_sputum",
        "rusty_sputum", "blood_in_sputum", "fast_heart_rate", "palpitations",
        "throat_irritation", "patches_in_throat",
    ],
    "Abdomen / Digestive": [
        "abdominal_pain", "stomach_pain", "belly_pain", "vomiting", "nausea",
        "diarrhoea", "constipation", "acidity", "indigestion", "distention_of_abdomen",
        "swelling_of_stomach", "stomach_bleeding", "passage_of_gases",
    ],
    "Limbs / Musculoskeletal": [
        "joint_pain", "knee_pain", "hip_joint_pain", "muscle_pain", "muscle_weakness",
        "muscle_wasting", "swelling_joints", "movement_stiffness", "painful_walking",
        "weakness_in_limbs", "cramps", "back_pain",
    ],
    "Skin": [
        "itching", "skin_rash", "nodal_skin_eruptions", "skin_peeling", "blister",
        "pus_filled_pimples", "blackheads", "bruising", "yellowish_skin", "dischromic_patches",
    ],
}

REGION_SHAPES = {
    "Head / Neurological": dict(x0=0.42, y0=0.85, x1=0.58, y1=1.0),
    "Chest / Respiratory": dict(x0=0.32, y0=0.55, x1=0.68, y1=0.82),
    "Abdomen / Digestive": dict(x0=0.35, y0=0.35, x1=0.65, y1=0.55),
    "Limbs / Musculoskeletal": dict(x0=0.15, y0=0.0, x1=0.85, y1=0.33),
    "Skin": dict(x0=0.05, y0=0.55, x1=0.15, y1=0.82),
}


def region_counts_from_symptoms(active_symptoms: set[str]) -> dict[str, int]:
    counts = {}
    for region, symptoms in BODY_REGION_SYMPTOM_MAP.items():
        counts[region] = len(active_symptoms.intersection(symptoms))
    return counts


def body_region_figure(active_symptoms: set[str] | None = None) -> go.Figure:
    active_symptoms = active_symptoms or set()
    counts = region_counts_from_symptoms(active_symptoms)
    max_count = max(counts.values()) if counts and max(counts.values()) > 0 else 1

    fig = go.Figure()
    for region, box in REGION_SHAPES.items():
        intensity = counts.get(region, 0) / max_count
        color = f"rgba(231,76,60,{0.15 + 0.65 * intensity})" if intensity > 0 else "rgba(52,152,219,0.15)"
        fig.add_shape(
            type="rect", x0=box["x0"], y0=box["y0"], x1=box["x1"], y1=box["y1"],
            line=dict(color="#2c3e50", width=1.5), fillcolor=color,
        )
        fig.add_annotation(
            x=(box["x0"] + box["x1"]) / 2, y=(box["y0"] + box["y1"]) / 2,
            text=f"{region}<br>({counts.get(region, 0)} active symptom areas)",
            showarrow=False, font=dict(size=10),
        )

    fig.update_layout(
        title="Schematic Body-Region Symptom Summary (illustrative only - not real anatomy)",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=600,
        plot_bgcolor="white",
    )
    return fig
