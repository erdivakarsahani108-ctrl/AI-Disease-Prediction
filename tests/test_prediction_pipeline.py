"""End-to-end hybrid prediction-pipeline tests (Top-K prediction, emergency
detection, low-confidence / uncertainty-aware handling, edge cases)."""

import pytest

from src.prediction.pipeline import run_pipeline


@pytest.fixture(autouse=True)
def _require_model():
    from src.utils.paths import BEST_MODEL_PKL

    if not BEST_MODEL_PKL.exists():
        pytest.skip("Model not trained yet - run `python scripts/train_model.py`.")


def test_normal_prediction_returns_top_k():
    r = run_pipeline("I have high fever, headache, joint pain and vomiting since 3 days.", top_k=5)
    assert r.status == "ok"
    assert 1 <= len(r.top_predictions) <= 5
    assert all(0 <= p["confidence"] <= 1 for p in r.top_predictions)


def test_top_predictions_sorted_descending():
    r = run_pipeline("I have high fever, headache, joint pain and vomiting since 3 days.", top_k=5)
    confidences = [p["confidence"] for p in r.top_predictions]
    assert confidences == sorted(confidences, reverse=True)


def test_explanation_present_for_ok_prediction():
    r = run_pipeline("I have high fever, headache, joint pain and vomiting since 3 days.")
    assert r.explanation is not None
    assert len(r.explanation["contributions"]) > 0


def test_emergency_detection_takes_priority():
    r = run_pipeline("I have severe chest pain and cant breathe.")
    assert r.status == "emergency"
    assert len(r.emergency_reasons) > 0


def test_emergency_status_has_no_prediction():
    r = run_pipeline("I have severe chest pain and cant breathe.")
    assert r.top_predictions == []


def test_low_confidence_single_vague_symptom_is_uncertain_or_low_conf():
    r = run_pipeline("fever")
    # Either explicitly insufficient, or a valid low-symptom-count prediction;
    # in both cases it must not silently crash and must expose completeness info.
    assert r.status in {"insufficient_info", "ok"}
    assert r.completeness_pct is not None


def test_no_symptoms_detected_returns_insufficient_info():
    r = run_pipeline("I feel sick.")
    assert r.status == "insufficient_info"


def test_empty_input_does_not_crash():
    r = run_pipeline("")
    assert r.status == "insufficient_info"


def test_negation_excludes_symptom_from_prediction_input():
    r = run_pipeline("I don't have cough, but I have fever and headache.")
    assert "cough" not in r.active_symptoms


def test_hindi_input_produces_prediction():
    r = run_pipeline("Mujhe bukhar, khansi aur sir dard hai, saath mein weakness bhi hai.")
    assert r.status == "ok"
    assert "high_fever" in r.active_symptoms


def test_conversation_state_accumulates_symptoms_across_turns():
    known = {"high_fever"}
    r = run_pipeline("I also have a headache.", known_symptoms=known)
    assert "high_fever" in r.active_symptoms
    assert "headache" in r.active_symptoms


def test_disease_info_present_for_ok_prediction():
    r = run_pipeline("I have high fever, headache, joint pain and vomiting since 3 days.")
    assert r.disease_info is not None
    assert "description" in r.disease_info
