"""Hybrid AI prediction pipeline - the central orchestrator combining NLP,
rule-based safety checks, the trained ML model, the disease knowledge base
and Explainable AI, exactly following the architecture required by the
project brief:

    User Message
        -> NLP (symptom extraction, negation, duration)
        -> Symptom Normalization (canonical vocabulary)
        -> Safety Check (red-flag / emergency detection - HIGHEST PRIORITY)
        -> ML Prediction (Top-1 / Top-3 / Top-5 + confidence)
        -> Uncertainty Check (insufficient info -> do not force a prediction)
        -> Disease Knowledge Base lookup
        -> Explainable AI (why this prediction)
        -> Risk Indicator
        -> Final structured Response
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.ml.explain import compare_top_predictions, explain_prediction
from src.nlp.extraction import ExtractionResult, extract_symptoms
from src.prediction.interview import select_next_question, information_completeness
from src.prediction.risk import assess_risk
from src.prediction.safety import EMERGENCY_MESSAGE, check_red_flags
from src.preprocessing.clean import build_disease_symptom_pool, load_real_disease_symptom_sets
from src.utils.paths import (
    BEST_MODEL_PKL,
    DISEASE_DICTIONARY_CSV,
    ENCODERS_PKL,
    MODEL_METADATA_JSON,
)

LOW_CONFIDENCE_THRESHOLD = 0.30
MIN_SYMPTOMS_FOR_PREDICTION = 1
TARGET_SYMPTOM_COUNT_FOR_FULL_COMPLETENESS = 5


@dataclass
class PredictionResponse:
    status: str  # "emergency" | "insufficient_info" | "ok"
    message: str = ""
    extraction: ExtractionResult | None = None
    active_symptoms: list = field(default_factory=list)
    top_predictions: list = field(default_factory=list)  # [{"disease":.., "confidence":..}, ...]
    explanation: dict | None = None
    comparison_table: pd.DataFrame | None = None
    risk: dict | None = None
    completeness_pct: int = 0
    next_question: dict | None = None
    disease_info: dict | None = None
    emergency_reasons: list = field(default_factory=list)


@lru_cache(maxsize=1)
def load_model_bundle():
    if not BEST_MODEL_PKL.exists():
        raise FileNotFoundError("Model not found. Run `python scripts/train_model.py` first.")
    model = joblib.load(BEST_MODEL_PKL)
    encoders = joblib.load(ENCODERS_PKL)
    metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8"))
    return {
        "model": model,
        "label_encoder": encoders["label_encoder"],
        "feature_columns": encoders["feature_columns"],
        "metadata": metadata,
    }


@lru_cache(maxsize=1)
def load_disease_pool():
    real_df = load_real_disease_symptom_sets()
    return build_disease_symptom_pool(real_df)


@lru_cache(maxsize=1)
def load_disease_dictionary() -> pd.DataFrame:
    return pd.read_csv(DISEASE_DICTIONARY_CSV)


def build_feature_vector(active_symptoms: set[str], feature_columns: list[str]) -> pd.DataFrame:
    row = {col: (1 if col in active_symptoms else 0) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def run_pipeline(
    text: str,
    known_symptoms: set[str] | None = None,
    denied_symptoms: set[str] | None = None,
    top_k: int = 5,
) -> PredictionResponse:
    """Run the full hybrid pipeline for one user turn.

    ``known_symptoms``/``denied_symptoms`` let the caller (e.g. the Streamlit
    chat UI) accumulate symptoms confirmed/denied across multiple turns of a
    conversation, not just the current message.
    """
    known_symptoms = set(known_symptoms or set())
    denied_symptoms = set(denied_symptoms or set())

    extraction = extract_symptoms(text)
    known_symptoms |= set(extraction.extracted_symptoms)
    denied_symptoms |= set(extraction.negated_symptoms)
    known_symptoms -= denied_symptoms

    # --- 1. SAFETY CHECK (highest priority) ---------------------------------
    safety = check_red_flags(known_symptoms, raw_text=text)
    if safety["triggered"]:
        return PredictionResponse(
            status="emergency",
            message=EMERGENCY_MESSAGE,
            extraction=extraction,
            active_symptoms=sorted(known_symptoms),
            emergency_reasons=safety["reasons"],
        )

    # --- 2. INSUFFICIENT INFORMATION CHECK ----------------------------------
    if len(known_symptoms) < MIN_SYMPTOMS_FOR_PREDICTION:
        return PredictionResponse(
            status="insufficient_info",
            message=(
                "I couldn't identify any clear symptoms in your message. "
                "Could you describe what you are experiencing (e.g. fever, "
                "headache, cough, duration)?"
            ),
            extraction=extraction,
            active_symptoms=sorted(known_symptoms),
            completeness_pct=information_completeness(known_symptoms, TARGET_SYMPTOM_COUNT_FOR_FULL_COMPLETENESS),
        )

    # --- 3. ML PREDICTION ----------------------------------------------------
    bundle = load_model_bundle()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]

    X = build_feature_vector(known_symptoms, feature_columns)
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1][:top_k]
    top_predictions = [
        {"disease": label_encoder.classes_[i], "confidence": float(proba[i])} for i in order
    ]
    top1_conf = top_predictions[0]["confidence"]

    completeness = information_completeness(known_symptoms, TARGET_SYMPTOM_COUNT_FOR_FULL_COMPLETENESS)

    # --- 4. UNCERTAINTY-AWARE CHECK ------------------------------------------
    if top1_conf < LOW_CONFIDENCE_THRESHOLD or len(known_symptoms) < 2:
        pool = load_disease_pool()
        next_q = select_next_question(
            [p["disease"] for p in top_predictions], known_symptoms, denied_symptoms, pool
        )
        return PredictionResponse(
            status="insufficient_info",
            message=(
                "Insufficient information for a reliable model prediction. "
                "Model confidence is low with the symptoms provided so far."
            ),
            extraction=extraction,
            active_symptoms=sorted(known_symptoms),
            top_predictions=top_predictions,
            completeness_pct=completeness,
            next_question=next_q,
        )

    # --- 5. EXPLAINABLE AI ----------------------------------------------------
    top1_index = int(order[0])
    explanation = explain_prediction(model, feature_columns, X, top1_index)
    comparison_table = compare_top_predictions(
        model,
        feature_columns,
        X,
        [int(i) for i in order],
        [p["disease"] for p in top_predictions],
    )

    # --- 6. RISK INDICATOR -----------------------------------------------------
    risk = assess_risk(known_symptoms, extraction.duration_days, top1_conf)

    # --- 7. KNOWLEDGE BASE LOOKUP -----------------------------------------------
    disease_df = load_disease_dictionary()
    top_disease_row = disease_df[disease_df["disease"] == top_predictions[0]["disease"]]
    disease_info = top_disease_row.iloc[0].to_dict() if not top_disease_row.empty else None

    # --- 8. NEXT QUESTION (keep refining if completeness is low) ---------------
    next_q = None
    if completeness < 100:
        pool = load_disease_pool()
        next_q = select_next_question(
            [p["disease"] for p in top_predictions], known_symptoms, denied_symptoms, pool
        )

    return PredictionResponse(
        status="ok",
        message="Prediction generated. Model confidence is not a medically validated probability.",
        extraction=extraction,
        active_symptoms=sorted(known_symptoms),
        top_predictions=top_predictions,
        explanation=explanation,
        comparison_table=comparison_table,
        risk=risk,
        completeness_pct=completeness,
        next_question=next_q,
        disease_info=disease_info,
    )
