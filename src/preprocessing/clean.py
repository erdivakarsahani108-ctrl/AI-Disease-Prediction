"""Core data-cleaning and integration logic for the disease-symptom dataset.

Pipeline implemented here (see docs/dataset_methodology.md for the full
narrative):

    Raw wide-format CSV (Disease, Symptom_1..Symptom_17)
        -> melt to long format (Disease, Symptom)
        -> clean/standardize every symptom token
        -> drop empty tokens
        -> re-pivot to (record -> set of symptoms) rows
        -> drop duplicate symptom-sets (this is the "real" deduplicated data)
        -> build wide binary one-hot matrix over the full symptom vocabulary
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.categories import get_category
from src.utils.paths import (
    RAW_DATASET_CSV,
    RAW_DESCRIPTION_CSV,
    RAW_PRECAUTION_CSV,
    RAW_SEVERITY_CSV,
)
from src.utils.text import clean_symptom_token

SYMPTOM_COLUMNS_PREFIX = "Symptom_"

# The public source files spell a couple of symptoms inconsistently between
# dataset.csv and Symptom-severity.csv (a known quirk of this dataset). We
# correct these known spelling variants to a single canonical token so the
# same real-world symptom is never split into two different features.
SYMPTOM_TOKEN_CORRECTIONS = {
    "foul_smell_of_urine": "foul_smell_ofurine",
    "spotting_urination": "spotting_urination",
    "dischromic_patches": "dischromic_patches",
}


def _canonical(token: str) -> str:
    return SYMPTOM_TOKEN_CORRECTIONS.get(token, token)


def load_symptom_vocabulary() -> pd.DataFrame:
    """Load the canonical symptom vocabulary + clinical severity weight.

    The source ``Symptom-severity.csv`` contains one stray row
    (``Symptom == "prognosis"``) that is a header artifact, not a real
    symptom - it is filtered out here.
    """
    sev = pd.read_csv(RAW_SEVERITY_CSV)
    sev["Symptom"] = sev["Symptom"].apply(clean_symptom_token).apply(_canonical)
    sev = sev[sev["Symptom"] != "prognosis"]
    sev = sev[sev["Symptom"] != ""]
    sev = sev.drop_duplicates(subset="Symptom").reset_index(drop=True)
    sev = sev.rename(columns={"weight": "severity_weight"})
    return sev


def load_disease_descriptions() -> pd.DataFrame:
    desc = pd.read_csv(RAW_DESCRIPTION_CSV)
    desc["Disease"] = desc["Disease"].str.strip()
    return desc


def load_disease_precautions() -> pd.DataFrame:
    prec = pd.read_csv(RAW_PRECAUTION_CSV)
    prec["Disease"] = prec["Disease"].str.strip()
    return prec


def load_real_disease_symptom_sets() -> pd.DataFrame:
    """Read the raw Kaggle-style wide CSV and return one row per UNIQUE
    (disease, frozenset-of-symptoms) combination found in the real data.

    Returns a DataFrame with columns: ``disease``, ``symptoms`` (sorted list).
    """
    raw = pd.read_csv(RAW_DATASET_CSV)
    symptom_cols = [c for c in raw.columns if c.startswith(SYMPTOM_COLUMNS_PREFIX)]

    records = []
    for _, row in raw.iterrows():
        disease = str(row["Disease"]).strip()
        symptoms = set()
        for col in symptom_cols:
            token = clean_symptom_token(row[col])
            token = _canonical(token)
            if token:
                symptoms.add(token)
        if symptoms:
            records.append((disease, tuple(sorted(symptoms))))

    df = pd.DataFrame(records, columns=["disease", "symptoms"])
    before = len(df)
    df = df.drop_duplicates(subset=["disease", "symptoms"]).reset_index(drop=True)
    after = len(df)
    df.attrs["raw_row_count"] = before + (len(raw) - before)  # informational
    df.attrs["duplicate_rows_removed"] = len(raw) - after
    return df


def build_disease_symptom_pool(real_df: pd.DataFrame) -> dict:
    """For every disease, compute the union of ALL symptoms that appear for
    it anywhere in the real (deduplicated) data. This pool is later used as
    the strict sampling universe for synthetic augmentation, guaranteeing
    that no symptom-disease association is ever fabricated.
    """
    pool: dict[str, set] = {}
    for disease, symptoms in zip(real_df["disease"], real_df["symptoms"]):
        pool.setdefault(disease, set()).update(symptoms)
    return {k: sorted(v) for k, v in pool.items()}


def attach_metadata(disease: str) -> dict:
    return {
        "disease_category": get_category(disease),
    }
