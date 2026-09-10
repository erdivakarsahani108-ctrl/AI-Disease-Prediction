"""Model loading & basic prediction sanity tests."""

import json

import joblib
import pandas as pd
import pytest

from src.utils.paths import BEST_MODEL_PKL, ENCODERS_PKL, MODEL_METADATA_JSON


@pytest.fixture(scope="module")
def bundle():
    if not BEST_MODEL_PKL.exists():
        pytest.skip("Model not trained yet - run `python scripts/train_model.py`.")
    model = joblib.load(BEST_MODEL_PKL)
    encoders = joblib.load(ENCODERS_PKL)
    metadata = json.loads(MODEL_METADATA_JSON.read_text(encoding="utf-8"))
    return model, encoders, metadata


def test_model_loads(bundle):
    model, encoders, metadata = bundle
    assert model is not None
    assert "label_encoder" in encoders
    assert "feature_columns" in encoders


def test_metadata_has_expected_keys(bundle):
    _, _, metadata = bundle
    for key in ["best_model_name", "metrics", "classes", "n_features"]:
        assert key in metadata


def test_metrics_are_reasonable(bundle):
    _, _, metadata = bundle
    m = metadata["metrics"]
    assert 0.5 <= m["accuracy"] <= 1.0
    assert 0.3 <= m["macro_f1"] <= 1.0


def test_prediction_shape(bundle):
    model, encoders, metadata = bundle
    feature_cols = encoders["feature_columns"]
    row = {c: 0 for c in feature_cols}
    if "high_fever" in row:
        row["high_fever"] = 1
    if "headache" in row:
        row["headache"] = 1
    X = pd.DataFrame([row], columns=feature_cols)
    proba = model.predict_proba(X)[0]
    assert len(proba) == len(metadata["classes"])
    assert abs(sum(proba) - 1.0) < 1e-6


def test_model_comparison_file_exists():
    from src.utils.paths import MODEL_COMPARISON_CSV

    assert MODEL_COMPARISON_CSV.exists()
    comp = pd.read_csv(MODEL_COMPARISON_CSV)
    assert len(comp) == 6  # 6 candidate algorithms compared
    assert "selection_score" in comp.columns
