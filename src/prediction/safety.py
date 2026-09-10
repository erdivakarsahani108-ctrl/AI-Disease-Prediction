"""Emergency / red-flag safety layer.

This layer runs BEFORE the ML prediction step and takes priority over it,
per the project's Responsible-AI requirement: potentially serious
("red-flag") presentations must never be masked by a routine disease
prediction.

Two detection mechanisms are combined:
    1. Structured red-flag SYMPTOM COMBINATIONS (from the trained vocabulary)
    2. Free-text EMERGENCY KEYWORD scan (English + Hindi/Hinglish) for
       phrases that may not map cleanly to a single vocabulary symptom
       (e.g. "can't breathe", "unconscious", "heavy bleeding").
"""

from __future__ import annotations

# Symptom keys (from the trained vocabulary) that are, on their own or in
# combination, potentially serious.
SINGLE_RED_FLAG_SYMPTOMS = {
    "coma",
    "altered_sensorium",
    "weakness_of_one_body_side",
    "slurred_speech",
    "stomach_bleeding",
    "blood_in_sputum",
    "acute_liver_failure",
}

RED_FLAG_COMBINATIONS = [
    ({"chest_pain", "breathlessness"}, "Chest pain with breathing difficulty"),
    ({"chest_pain", "fast_heart_rate"}, "Chest pain with rapid heartbeat"),
    ({"breathlessness", "fast_heart_rate"}, "Severe breathlessness with rapid heartbeat"),
    ({"high_fever", "stiff_neck"}, "High fever with stiff neck"),
    ({"vomiting", "stomach_bleeding"}, "Vomiting with internal bleeding signs"),
]

EMERGENCY_TEXT_KEYWORDS = [
    # English
    "can't breathe", "cant breathe", "cannot breathe", "not breathing",
    "unconscious", "unresponsive", "severe bleeding", "heavy bleeding",
    "heart attack", "stroke", "seizure", "convulsion", "suicidal",
    "severe chest pain", "crushing chest pain", "coughing blood",
    "passed out", "collapsed",
    # Hindi / Hinglish
    "saans nahi aa rahi", "behosh", "behoshi", "bahut khoon", "dora pad gaya",
    "dil ka dora", "khoon ki ulti",
]


def check_red_flags(active_symptoms: set[str], raw_text: str = "") -> dict:
    """Return {"triggered": bool, "reasons": [str, ...]}"""
    reasons = []

    for symptom in active_symptoms:
        if symptom in SINGLE_RED_FLAG_SYMPTOMS:
            reasons.append(f"Potentially serious symptom reported: {symptom.replace('_', ' ').title()}")

    for combo, label in RED_FLAG_COMBINATIONS:
        if combo.issubset(active_symptoms):
            reasons.append(f"Potentially serious symptom combination: {label}")

    text_lower = (raw_text or "").lower()
    for kw in EMERGENCY_TEXT_KEYWORDS:
        if kw in text_lower:
            reasons.append(f"Emergency phrase detected in message: \"{kw}\"")

    return {"triggered": bool(reasons), "reasons": reasons}


EMERGENCY_MESSAGE = (
    "🚨 Potentially serious symptoms detected.\n\n"
    "This AI assistant recommends **immediate professional / emergency medical "
    "evaluation**. Please contact a doctor, visit the nearest emergency "
    "department, or call your local emergency number right away.\n\n"
    "This tool cannot assess emergencies and must not delay professional care."
)
