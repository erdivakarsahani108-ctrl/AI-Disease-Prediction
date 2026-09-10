"""Intelligent symptom-interview question selection.

Given the symptoms confirmed so far and the current ML top-K candidate
diseases, selects the next most useful follow-up question: a symptom that
(a) is strongly associated with the current top candidates, (b) has not
already been confirmed/denied by the user, prioritized by how well it
differentiates between the top candidates and by clinical severity weight.
"""

from __future__ import annotations

import json

import pandas as pd

from src.prediction.safety import SINGLE_RED_FLAG_SYMPTOMS
from src.utils.paths import DISEASE_DICTIONARY_CSV, SYMPTOM_VOCAB_JSON
from src.utils.text import to_display_name


def _severity_weights() -> dict:
    if SYMPTOM_VOCAB_JSON.exists():
        return json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8")).get("severity_weight", {})
    return {}


def select_next_question(
    top_candidate_diseases: list[str],
    known_symptoms: set[str],
    denied_symptoms: set[str],
    disease_symptom_pool: dict[str, list[str]],
    max_candidates_considered: int = 3,
) -> dict | None:
    """Return {"symptom": key, "question": str} or None if nothing useful to ask."""
    weights = _severity_weights()
    considered = top_candidate_diseases[:max_candidates_considered]

    candidate_symptoms: dict[str, int] = {}
    for disease in considered:
        for s in disease_symptom_pool.get(disease, []):
            if s in known_symptoms or s in denied_symptoms:
                continue
            if s in SINGLE_RED_FLAG_SYMPTOMS:
                # Red-flag / emergency-adjacent symptoms are handled by the
                # dedicated safety layer, not casually asked about here.
                continue
            candidate_symptoms[s] = candidate_symptoms.get(s, 0) + 1

    if not candidate_symptoms:
        return None

    # Prefer symptoms that appear in SOME but not ALL top candidates (best
    # differentiators), then by clinical severity weight.
    def score(item):
        symptom, count = item
        differentiation = 1 if 0 < count < len(considered) else 0
        return (differentiation, weights.get(symptom, 1))

    best_symptom = max(candidate_symptoms.items(), key=score)[0]
    display = to_display_name(best_symptom)
    return {
        "symptom": best_symptom,
        "question": f"Do you also have {display.lower()}?",
    }


def information_completeness(known_symptoms: set[str], target_symptom_count: int = 5) -> int:
    """Simple, transparent completeness percentage used by the chat UI."""
    pct = int(min(100, round(len(known_symptoms) / target_symptom_count * 100)))
    return pct
