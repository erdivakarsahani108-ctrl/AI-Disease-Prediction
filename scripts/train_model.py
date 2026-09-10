"""
scripts/train_model.py
========================
Reproducible Data-Science pipeline step 2: MODEL TRAINING & COMPARISON.

Trains and compares 6 classical ML algorithms on the disease-symptom
dataset, evaluates each with cross-validation + held-out test metrics,
selects the best model using a WEIGHTED combination of metrics (not
accuracy alone, per project requirement), and persists:

    models/best_model.pkl
    models/encoders.pkl
    models/model_metadata.json
    models/model_comparison.csv
    models/experiments_log.csv   (appended)

Run:
    python scripts/train_model.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.ml.data_loader import get_feature_columns, load_dataset, train_test_split_dataset
from src.ml.drift import build_drift_baseline
from src.ml.experiment_tracker import log_experiment
from src.utils.paths import (
    BEST_MODEL_PKL,
    ENCODERS_PKL,
    FINAL_DATASET_CSV,
    MODEL_COMPARISON_CSV,
    MODEL_METADATA_JSON,
    RANDOM_SEED,
)

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        n_jobs=-1,
    ),
}

# Weighted selection score: prioritizes macro metrics (fair across all
# diseases, incl. rare ones) over raw accuracy, plus a stability bonus for
# cross-validation consistency.
SELECTION_WEIGHTS = {
    "macro_f1": 0.40,
    "weighted_f1": 0.20,
    "macro_recall": 0.20,
    "cv_mean_f1_macro": 0.20,
}


def dataset_version_hash() -> str:
    data = FINAL_DATASET_CSV.read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def evaluate_model(name, model, X_train, X_test, y_train, y_test, n_classes):
    min_class_count = min(np.bincount(y_train))
    cv_folds = max(2, min(5, min_class_count))

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    except Exception:
        cv_scores = np.array([np.nan])

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cv_folds": cv_folds,
        "cv_mean_f1_macro": float(np.nanmean(cv_scores)),
        "cv_std_f1_macro": float(np.nanstd(cv_scores)),
        "train_time_sec": round(train_time, 4),
        "predict_time_sec": round(predict_time, 4),
    }
    return model, metrics


def main():
    print("Loading processed dataset ...")
    df = load_dataset()
    feature_cols = get_feature_columns(df)
    X_train, X_test, y_train, y_test, label_encoder, feature_cols = train_test_split_dataset(df)
    n_classes = len(label_encoder.classes_)
    version = dataset_version_hash()

    print(f"Dataset: {len(df)} records | {len(feature_cols)} features | {n_classes} diseases")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)} (stratified, seed={RANDOM_SEED})")

    results = []
    fitted_models = {}
    for name, model in CANDIDATE_MODELS.items():
        print(f"\nTraining: {name} ...")
        fitted, metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test, n_classes)
        fitted_models[name] = fitted
        row = {"model": name, **metrics}
        results.append(row)
        log_experiment(
            model_name=name,
            dataset_version=version,
            n_features=len(feature_cols),
            hyperparameters=fitted.get_params(),
            metrics=metrics,
        )
        print(
            f"  accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} cv_f1_macro={metrics['cv_mean_f1_macro']:.4f}"
        )

    comparison_df = pd.DataFrame(results)

    # Weighted composite selection score (NOT accuracy-only selection).
    comparison_df["selection_score"] = (
        comparison_df["macro_f1"] * SELECTION_WEIGHTS["macro_f1"]
        + comparison_df["weighted_f1"] * SELECTION_WEIGHTS["weighted_f1"]
        + comparison_df["macro_recall"] * SELECTION_WEIGHTS["macro_recall"]
        + comparison_df["cv_mean_f1_macro"] * SELECTION_WEIGHTS["cv_mean_f1_macro"]
    )
    comparison_df = comparison_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"\nSaved model comparison -> {MODEL_COMPARISON_CSV}")

    best_row = comparison_df.iloc[0]
    best_name = best_row["model"]
    best_model = fitted_models[best_name]
    print(f"\n{'=' * 60}\nBEST MODEL SELECTED: {best_name} (selection_score={best_row['selection_score']:.4f})\n{'=' * 60}")

    BEST_MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PKL)

    encoders_payload = {
        "label_encoder": label_encoder,
        "feature_columns": feature_cols,
    }
    joblib.dump(encoders_payload, ENCODERS_PKL)

    metadata = {
        "best_model_name": best_name,
        "dataset_version_hash": version,
        "n_records": len(df),
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "classes": label_encoder.classes_.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_seed": RANDOM_SEED,
        "selection_weights": SELECTION_WEIGHTS,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best_row.items()},
        "all_models_compared": comparison_df["model"].tolist(),
        "trained_at": pd.Timestamp.now().isoformat(),
        "notes": (
            "Model selected via weighted composite of macro-F1, weighted-F1, "
            "macro-recall and mean cross-validated F1-macro (not accuracy alone), "
            "because the dataset has real class imbalance (see docs/data_quality_report.md)."
        ),
    }
    MODEL_METADATA_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {BEST_MODEL_PKL.name}, {ENCODERS_PKL.name}, {MODEL_METADATA_JSON.name}")

    build_drift_baseline(df, feature_cols)
    print("Saved drift baseline -> models/drift_baseline.json")

    print("\nModel comparison summary:")
    print(comparison_df[["model", "accuracy", "macro_f1", "weighted_f1", "cv_mean_f1_macro", "selection_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
