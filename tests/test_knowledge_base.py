"""Knowledge-base tests: Disease Dictionary and Medicine Information."""

import pandas as pd

from src.utils.paths import DISEASE_DICTIONARY_CSV, MEDICINE_DICTIONARY_CSV, SYMPTOM_DICTIONARY_CSV


def test_disease_dictionary_loads_and_has_required_columns():
    df = pd.read_csv(DISEASE_DICTIONARY_CSV)
    required = {
        "disease", "category", "description", "common_symptoms",
        "general_precautions", "risk_factors", "when_to_seek_professional_care",
        "similar_diseases", "source", "verification_date",
    }
    assert required.issubset(set(df.columns))
    assert len(df) >= 30


def test_disease_search_by_name():
    df = pd.read_csv(DISEASE_DICTIONARY_CSV)
    result = df[df["disease"].str.contains("Malaria", case=False)]
    assert len(result) == 1


def test_disease_search_by_category():
    df = pd.read_csv(DISEASE_DICTIONARY_CSV)
    result = df[df["category"] == "Respiratory"]
    assert len(result) > 0


def test_every_disease_has_a_description():
    df = pd.read_csv(DISEASE_DICTIONARY_CSV)
    assert df["description"].isna().sum() == 0
    assert (df["description"].str.len() > 0).all()


def test_symptom_dictionary_loads():
    df = pd.read_csv(SYMPTOM_DICTIONARY_CSV)
    assert len(df) >= 100
    assert {"symptom", "display_name", "severity_weight", "severity_band"}.issubset(df.columns)


def test_medicine_dictionary_loads_without_parser_errors():
    df = pd.read_csv(MEDICINE_DICTIONARY_CSV)
    required = {
        "medicine_name", "generic_name", "drug_class", "common_uses",
        "precautions", "common_side_effects", "important_warnings", "source", "verification_date",
    }
    assert required.issubset(set(df.columns))
    assert len(df) > 0


def test_medicine_search_by_name():
    df = pd.read_csv(MEDICINE_DICTIONARY_CSV)
    result = df[df["medicine_name"].str.contains("Paracetamol", case=False)]
    assert len(result) == 1


def test_medicine_dictionary_has_no_dosage_prescription_language():
    """Guard against accidental prescriptive dosage instructions (the
    project explicitly forbids auto-prescription)."""
    df = pd.read_csv(MEDICINE_DICTIONARY_CSV)
    combined = " ".join(df["common_uses"].astype(str)).lower()
    assert "mg twice daily" not in combined
    assert "take 2 tablets" not in combined
