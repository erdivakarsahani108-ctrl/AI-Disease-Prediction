# Data Sources

## Primary dataset (real, public)

- **Name:** Disease-Symptom Prediction dataset (commonly distributed as
  `dataset.csv`, `Symptom-severity.csv`, `symptom_Description.csv`,
  `symptom_precaution.csv`)
- **Original provenance:** Widely circulated academic/Kaggle dataset
  (originally associated with the "itachi9604/disease-symptom-description-
  dataset" Kaggle listing); this project retrieved a public GitHub mirror
  at build time: `https://github.com/hemasriram111/Disease-Symptom-Dataset`.
- **License / usage:** Publicly available for research/educational reuse;
  no personally identifiable patient data is present (it is a symptom-
  association reference table, not patient records).
- **Contents used by this project:**
  - `dataset.csv` - 4,920 raw rows, 41 diseases, up to 17 symptom slots per
    row -> reduced to **304 unique real disease-symptom combinations**
    after deduplication (see `docs/dataset_methodology.md`).
  - `Symptom-severity.csv` - 131 valid symptoms with a clinical severity
    weight (1-7) used for risk-indicator scoring and Explainable AI
    fallback ranking.
  - `symptom_Description.csv` - short description per disease (41 rows),
    used verbatim in the Disease Dictionary.
  - `symptom_precaution.csv` - 4 general precautions per disease (41 rows),
    used verbatim in the Disease Dictionary.
- **Local copies:** stored under `data/raw/` for full reproducibility.

## Medicine Information (curated seed, NOT a licensed dataset)

No free, machine-readable, licensed medicine/drug-interaction API or
dataset was available to this project. Per the project's "do not
fabricate, document the requirement" instruction:

- `data/medicine_dictionary.csv` is a **small, hand-curated set of 15
  extremely well-known, textbook-level generic medicines**, built from
  general public pharmacology knowledge consistent with the WHO Model
  List of Essential Medicines (a public reference list of internationally
  recognized essential medicines).
- Only high-level, non-dosage information is included (drug class, common
  uses, general precautions, common side effects, warnings) - deliberately
  excluding any specific dosage/prescription instructions.
- **Future Scope:** integrate a licensed structured medicine database or
  API (e.g. RxNorm, openFDA Drug Label API, DrugBank) for comprehensive,
  continuously-updated coverage. The `data/medicine_dictionary.csv` schema
  is designed so a future import script can simply replace/extend this
  file without any application code changes.

## Disease category taxonomy (curated, organizational only)

`src/preprocessing/categories.py` maps each of the 41 disease labels to a
standard body-system category (e.g. Respiratory, Gastrointestinal). This
is a purely organizational classification for filtering/grouping in the
UI - it does not add or alter any clinical fact.

## What was NOT fabricated

- No age, gender, severity, or other clinical metadata was invented for
  any record (the source dataset does not include these fields, and none
  were added).
- No new disease-symptom association was invented; synthetic records only
  resample subsets of symptoms already validated for that disease in the
  real data (see `docs/dataset_methodology.md`).
- No medicine dosage or prescription-specific fact is included anywhere in
  the application.
