"""Dataset validation tests."""

import pandas as pd
import pytest

from src.ml.data_loader import get_feature_columns, load_dataset
from src.utils.paths import FINAL_DATASET_CSV, SYMPTOM_VOCAB_JSON


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def test_dataset_file_exists():
    assert FINAL_DATASET_CSV.exists(), "Run `python scripts/prepare_data.py` first."


def test_dataset_has_minimum_size(df):
    assert len(df) >= 5000, "Dataset should contain a substantial number of records."


def test_dataset_has_required_columns(df):
    required = {"record_id", "disease", "disease_category", "symptoms", "source", "synthetic_flag"}
    assert required.issubset(set(df.columns))


def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_no_fully_duplicate_rows(df):
    feature_cols = get_feature_columns(df)
    assert df.duplicated(subset=feature_cols + ["disease"]).sum() == 0


def test_every_record_has_at_least_one_symptom(df):
    feature_cols = get_feature_columns(df)
    assert (df[feature_cols].sum(axis=1) > 0).all()


def test_synthetic_flag_is_boolean(df):
    assert df["synthetic_flag"].dtype == bool


def test_real_and_synthetic_both_present(df):
    assert (~df["synthetic_flag"]).sum() > 0, "There should be real records."
    assert df["synthetic_flag"].sum() > 0, "There should be synthetic records."


def test_multiple_diseases_present(df):
    assert df["disease"].nunique() >= 10


def test_symptom_vocabulary_matches_dataset_columns(df):
    import json

    vocab = json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8"))["symptoms"]
    feature_cols = get_feature_columns(df)
    assert set(vocab) == set(feature_cols)
