"""Lightweight, dependency-free experiment tracking.

Every training run appends one row per model to
``models/experiments_log.csv`` with model name, dataset size/version,
hyperparameters, and evaluation metrics + a timestamp, satisfying the
project's "Experiment Tracking" requirement without requiring a heavyweight
tracking server.

Future Scope: this CSV-based tracker can be swapped for MLflow by pointing
`log_experiment` at `mlflow.log_metrics/log_params` instead - the calling
code in scripts/train_model.py does not need to change.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from src.utils.paths import EXPERIMENTS_LOG_CSV


def log_experiment(
    model_name: str,
    dataset_version: str,
    n_features: int,
    hyperparameters: dict,
    metrics: dict,
) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "dataset_version": dataset_version,
        "n_features": n_features,
        "hyperparameters": json.dumps(hyperparameters, default=str),
        **metrics,
    }
    df_row = pd.DataFrame([row])
    if EXPERIMENTS_LOG_CSV.exists():
        df_row.to_csv(EXPERIMENTS_LOG_CSV, mode="a", header=False, index=False)
    else:
        EXPERIMENTS_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_row.to_csv(EXPERIMENTS_LOG_CSV, mode="w", header=True, index=False)


def load_experiments() -> pd.DataFrame:
    if not EXPERIMENTS_LOG_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_LOG_CSV)
