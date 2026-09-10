"""Personalized (AI-assisted, non-clinical) risk indicator.

Combines: number of active symptoms, clinical severity-weight of those
symptoms (from the public Symptom-severity.csv reference), reported
duration, and model confidence into a simple, transparent 🟢/🟡/🔴 banding.

This is explicitly an AI-assisted heuristic indicator, NOT a clinical risk
score - it is not derived from any validated clinical risk model, since age/
sex/comorbidity data are not part of this project's dataset (see
docs/limitations.md).
"""

from __future__ import annotations

import json

from src.utils.paths import SYMPTOM_VOCAB_JSON

_weights_cache = None


def _severity_weights() -> dict:
    global _weights_cache
    if _weights_cache is None:
        if SYMPTOM_VOCAB_JSON.exists():
            _weights_cache = json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8")).get(
                "severity_weight", {}
            )
        else:
            _weights_cache = {}
    return _weights_cache


def assess_risk(active_symptoms: set[str], duration_days: int | None, top1_confidence: float) -> dict:
    weights = _severity_weights()
    total_severity = sum(weights.get(s, 1) for s in active_symptoms)
    avg_severity = total_severity / len(active_symptoms) if active_symptoms else 0

    score = 0
    score += min(3, len(active_symptoms) // 2)  # more symptoms -> higher concern
    score += 2 if avg_severity >= 6 else (1 if avg_severity >= 4 else 0)
    score += 1 if (duration_days is not None and duration_days >= 7) else 0
    score += 1 if top1_confidence >= 0.7 else 0

    if score >= 5:
        band, emoji = "High Risk", "🔴"
    elif score >= 3:
        band, emoji = "Moderate Risk", "🟡"
    else:
        band, emoji = "Low Risk", "🟢"

    return {
        "band": band,
        "emoji": emoji,
        "score": score,
        "avg_symptom_severity": round(avg_severity, 2),
        "disclaimer": (
            "This is an AI-assisted risk indicator based on symptom count, "
            "reference clinical severity weights and duration - it is NOT a "
            "clinical diagnosis or validated medical risk score."
        ),
    }
