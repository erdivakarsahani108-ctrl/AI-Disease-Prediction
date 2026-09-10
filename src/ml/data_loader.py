"""Dataset loading helpers shared by training / evaluation / prediction code."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.paths import FINAL_DATASET_CSV, RANDOM_SEED, SYMPTOM_VOCAB_JSON

NON_FEATURE_COLUMNS = {
    "record_id",
    "disease",
    "disease_category",
    "symptoms",
    "symptom_count",
    "source",
    "synthetic_flag",
}


def load_symptom_columns() -> list[str]:
    payload = json.loads(SYMPTOM_VOCAB_JSON.read_text(encoding="utf-8"))
    return payload["symptoms"]


def load_dataset() -> pd.DataFrame:
    if not FINAL_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{FINAL_DATASET_CSV} not found. Run `python scripts/prepare_data.py` first."
        )
    return pd.read_csv(FINAL_DATASET_CSV)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def train_test_split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_SEED):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y_raw = df["disease"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, label_encoder, feature_cols
