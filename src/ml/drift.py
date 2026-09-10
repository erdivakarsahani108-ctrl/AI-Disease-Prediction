"""Data/Model drift detection.

Compares the symptom-frequency distribution of NEW prediction inputs
(logged over time) against a saved TRAINING baseline distribution, using a
simple, transparent Population Stability Index (PSI)-style measure per
symptom, aggregated into an overall drift verdict.

This is intentionally simple and explainable (no heavyweight statistical
dependencies) so it is easy to defend academically. It explicitly does
NOT claim drift automatically means the model has failed - it only flags
that the input distribution looks different from training, which may
warrant human review.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.paths import DRIFT_BASELINE_JSON


def build_drift_baseline(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Compute and persist the training-time symptom frequency baseline."""
    freqs = {col: float(df[col].mean()) for col in feature_columns}
    disease_freqs = df["disease"].value_counts(normalize=True).to_dict()
    baseline = {"symptom_frequency": freqs, "disease_frequency": disease_freqs, "n_records": len(df)}
    DRIFT_BASELINE_JSON.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def _load_baseline() -> dict | None:
    if not DRIFT_BASELINE_JSON.exists():
        return None
    return json.loads(DRIFT_BASELINE_JSON.read_text(encoding="utf-8"))


def compute_drift(recent_inputs: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Compare recent logged prediction inputs to the training baseline.

    ``recent_inputs`` is a DataFrame with one row per past prediction request
    (one-hot symptom columns matching ``feature_columns``).
    """
    baseline = _load_baseline()
    if baseline is None:
        return {"status": "no_baseline", "message": "Baseline not found - run scripts/train_model.py first."}

    if len(recent_inputs) < 5:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(recent_inputs)} logged prediction(s) - need at least 5 for a meaningful drift estimate.",
        }

    per_symptom_drift = {}
    for col in feature_columns:
        base_p = baseline["symptom_frequency"].get(col, 0.0001)
        base_p = max(base_p, 0.0001)
        recent_p = max(recent_inputs[col].mean(), 0.0001)
        psi = (recent_p - base_p) * np.log(recent_p / base_p)
        per_symptom_drift[col] = float(psi)

    total_psi = float(np.sum(list(per_symptom_drift.values())))

    if total_psi < 0.1:
        verdict = "Stable"
        emoji = "🟢"
    elif total_psi < 0.25:
        verdict = "Moderate Drift"
        emoji = "🟡"
    else:
        verdict = "Significant Drift"
        emoji = "🔴"

    top_shifted = sorted(per_symptom_drift.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "status": "ok",
        "verdict": verdict,
        "emoji": emoji,
        "total_psi": round(total_psi, 4),
        "top_shifted_symptoms": top_shifted,
        "n_recent_inputs": len(recent_inputs),
        "disclaimer": (
            "Drift indicates the recent input distribution differs from the "
            "training distribution. It does NOT automatically mean the model "
            "has failed - it is a signal that may warrant human review."
        ),
    }
