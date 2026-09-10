"""Lightweight local feedback storage.

Design choices driven by the project's Privacy requirements:
    - No names, phone numbers, addresses or other personal identifiers are
      collected or stored.
    - Feedback is stored ONLY for the current local session/deployment
      (a simple append-only CSV) - it is not sent anywhere else.
    - The model is NEVER automatically retrained from this feedback (per
      the Responsible-AI requirement) - it exists purely for qualitative
      review by the project maintainer.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.utils.paths import FEEDBACK_CSV

FEEDBACK_COLUMNS = [
    "timestamp",
    "predicted_disease",
    "confidence",
    "helpful",
    "confirmed_by_professional",
    "symptom_count",
]


def save_feedback(
    predicted_disease: str,
    confidence: float,
    helpful: bool,
    confirmed_by_professional: str = "Not specified",
    symptom_count: int = 0,
) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted_disease": predicted_disease,
        "confidence": round(float(confidence), 4),
        "helpful": bool(helpful),
        "confirmed_by_professional": confirmed_by_professional,
        "symptom_count": symptom_count,
    }
    df_row = pd.DataFrame([row], columns=FEEDBACK_COLUMNS)
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_CSV.exists():
        df_row.to_csv(FEEDBACK_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(FEEDBACK_CSV, mode="w", header=True, index=False)


def load_feedback() -> pd.DataFrame:
    if not FEEDBACK_CSV.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(FEEDBACK_CSV)


def feedback_summary() -> dict:
    df = load_feedback()
    if df.empty:
        return {"total": 0, "helpful_pct": None}
    return {
        "total": len(df),
        "helpful_pct": round(df["helpful"].astype(bool).mean() * 100, 1),
        "professional_confirmed_count": int((df["confirmed_by_professional"] == "Yes").sum()),
    }
