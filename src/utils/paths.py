"""Centralized filesystem paths for the AI Disease Prediction project.

Keeping every path in one place makes the project reproducible: scripts,
notebooks, the Streamlit app and the test-suite all import from here instead
of hard-coding relative paths that break depending on the current working
directory.
"""

from pathlib import Path

# Project root = two levels above this file (src/utils/paths.py -> project root)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DOCS_DIR = ROOT_DIR / "docs"
ASSETS_DIR = ROOT_DIR / "assets"
REPORTS_DIR = ROOT_DIR / "reports"
FEEDBACK_DIR = ROOT_DIR / "feedback"

# Raw source files (public dataset - see docs/data_sources.md)
RAW_DATASET_CSV = RAW_DATA_DIR / "dataset.csv"
RAW_SEVERITY_CSV = RAW_DATA_DIR / "Symptom-severity.csv"
RAW_PRECAUTION_CSV = RAW_DATA_DIR / "symptom_precaution.csv"
RAW_DESCRIPTION_CSV = RAW_DATA_DIR / "symptom_Description.csv"

# Processed / final artifacts
FINAL_DATASET_CSV = PROCESSED_DATA_DIR / "disease_dataset.csv"
SYMPTOM_VOCAB_JSON = PROCESSED_DATA_DIR / "symptom_vocabulary.json"
DATA_DICTIONARY_MD = DOCS_DIR / "data_dictionary.md"

# Knowledge base
DISEASE_DICTIONARY_CSV = DATA_DIR / "disease_dictionary.csv"
SYMPTOM_DICTIONARY_CSV = DATA_DIR / "symptom_dictionary.csv"
MEDICINE_DICTIONARY_CSV = DATA_DIR / "medicine_dictionary.csv"

# Model artifacts
BEST_MODEL_PKL = MODELS_DIR / "best_model.pkl"
ENCODERS_PKL = MODELS_DIR / "encoders.pkl"
MODEL_METADATA_JSON = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_CSV = MODELS_DIR / "model_comparison.csv"

# Experiment tracking / drift / feedback
EXPERIMENTS_LOG_CSV = MODELS_DIR / "experiments_log.csv"
DRIFT_BASELINE_JSON = MODELS_DIR / "drift_baseline.json"
FEEDBACK_CSV = FEEDBACK_DIR / "feedback_log.csv"

RANDOM_SEED = 42

for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FEEDBACK_DIR):
    _d.mkdir(parents=True, exist_ok=True)
