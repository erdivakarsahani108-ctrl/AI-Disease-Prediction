"""NLP symptom-extraction tests: English, Hindi, Hinglish, negation, edge
cases (per the project's testing requirements)."""

from src.nlp.extraction import extract_symptoms
from src.utils.text import clean_symptom_token, normalize_free_text, to_display_name


def test_clean_symptom_token_basic():
    assert clean_symptom_token(" High  Fever ") == "high_fever"
    assert clean_symptom_token("foul-smell of urine") == "foul_smell_of_urine"


def test_clean_symptom_token_handles_nan():
    assert clean_symptom_token(float("nan")) == ""
    assert clean_symptom_token(None) == ""


def test_to_display_name():
    assert to_display_name("high_fever") == "High Fever"


def test_english_multi_symptom_extraction():
    r = extract_symptoms("I have fever, cough and body pain.")
    assert "high_fever" in r.extracted_symptoms
    assert "cough" in r.extracted_symptoms


def test_hindi_extraction():
    r = extract_symptoms("Mujhe bukhar aur khansi hai.")
    assert "high_fever" in r.extracted_symptoms
    assert "cough" in r.extracted_symptoms


def test_hinglish_with_duration():
    r = extract_symptoms("Mujhe 3 din se bukhar, headache aur weakness hai.")
    assert "high_fever" in r.extracted_symptoms
    assert "headache" in r.extracted_symptoms
    assert "fatigue" in r.extracted_symptoms
    assert r.duration_days == 3


def test_negation_english():
    r = extract_symptoms("I don't have cough.")
    assert "cough" not in r.extracted_symptoms
    assert "cough" in r.negated_symptoms


def test_negation_english_no_apostrophe():
    r = extract_symptoms("I dont have cough.")
    assert "cough" not in r.extracted_symptoms


def test_single_word_symptom():
    r = extract_symptoms("fever")
    assert "high_fever" in r.extracted_symptoms


def test_symptoms_with_plus_separator():
    r = extract_symptoms("fever + headache + cough")
    assert {"high_fever", "headache", "cough"}.issubset(set(r.extracted_symptoms))


def test_vague_input_extracts_nothing():
    r = extract_symptoms("I feel sick.")
    assert r.extracted_symptoms == []


def test_empty_input():
    r = extract_symptoms("")
    assert r.extracted_symptoms == []
    assert r.duration_days is None


def test_long_input_does_not_crash():
    text = "I have fever and headache. " * 200
    r = extract_symptoms(text)
    assert "high_fever" in r.extracted_symptoms


def test_conflicting_symptoms_negation_wins():
    r = extract_symptoms("I have fever but I don't have fever anymore.")
    # The symptom is mentioned in both a positive and negative context;
    # negation must take precedence (fail-safe, non-fabrication behavior).
    assert "high_fever" not in r.extracted_symptoms


def test_duration_since_pattern():
    r = extract_symptoms("Since 2 days I have fever.")
    assert r.duration_days == 2


def test_normalize_free_text_collapses_whitespace():
    assert normalize_free_text("  Fever   AND   cough  ") == "fever and cough"


def test_all_synonym_keys_are_valid_vocabulary_symptoms():
    """Regression test: guards against synonym-dictionary keys that don't
    correspond to a real trained-model feature (previously caused a bug
    where 'body_pain' was extracted but had no matching model feature)."""
    import json

    from src.nlp.symptom_synonyms import SYMPTOM_SYNONYMS
    from src.utils.paths import SYMPTOM_VOCAB_JSON

    vocab = set(json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8"))["symptoms"])
    assert set(SYMPTOM_SYNONYMS.keys()).issubset(vocab)
