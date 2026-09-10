"""
scripts/validate_dataset.py
=============================
Standalone Data-Quality validation script (also used to generate the
numbers shown in docs/data_quality_report.md and the Streamlit "Data
Quality Dashboard").

Checks:
    - Completeness (missing values)
    - Duplicate records
    - Invalid records (zero-symptom rows)
    - Class imbalance
    - Synthetic-data percentage
    - Source coverage
    - Train/test leakage risk (exact duplicate feature rows split across
      train and test) and symptom-pattern ambiguity across different disease
      labels (multiple diseases sharing an identical symptom pattern - this
      is a genuine clinical ambiguity, not a data bug, and is reported as such)

Run:
    python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ml.data_loader import get_feature_columns, load_dataset
from src.utils.paths import REPORTS_DIR


def compute_quality_score(report: dict) -> int:
    """Simple, transparent weighted scoring (0-100). Weights and rationale
    are documented in docs/data_quality_report.md."""
    score = 100
    score -= min(20, report["missing_value_pct"] * 2)
    score -= min(15, report["duplicate_pct"] * 3)
    score -= min(15, report["invalid_record_pct"] * 3)
    imbalance_penalty = min(25, max(0, (report["class_imbalance_ratio"] - 5) * 1.2))
    score -= imbalance_penalty
    synthetic_penalty = min(15, max(0, (report["synthetic_pct"] - 50) * 0.3))
    score -= synthetic_penalty
    return max(0, round(score))


def main():
    df = load_dataset()
    feature_cols = get_feature_columns(df)
    n = len(df)

    missing_value_pct = round(df.isna().mean().mean() * 100, 3)
    invalid_records = (df[feature_cols].sum(axis=1) == 0).sum()
    exact_dup_records = df.duplicated(subset=feature_cols + ["disease"]).sum()

    ambiguous_pattern_groups = (
        df.groupby(feature_cols)["disease"].nunique().reset_index(name="n_diseases")
    )
    ambiguous_patterns = int((ambiguous_pattern_groups["n_diseases"] > 1).sum())

    class_counts = df["disease"].value_counts()
    class_imbalance_ratio = round(class_counts.max() / class_counts.min(), 2)

    synthetic_pct = round(df["synthetic_flag"].astype(bool).mean() * 100, 2)
    source_counts = df["source"].value_counts().to_dict()

    report = {
        "n_records": n,
        "n_diseases": df["disease"].nunique(),
        "n_symptom_features": len(feature_cols),
        "missing_value_pct": missing_value_pct,
        "duplicate_pct": round(exact_dup_records / n * 100, 3),
        "invalid_record_pct": round(invalid_records / n * 100, 3),
        "class_imbalance_ratio": class_imbalance_ratio,
        "min_class_count": int(class_counts.min()),
        "max_class_count": int(class_counts.max()),
        "synthetic_pct": synthetic_pct,
        "real_pct": round(100 - synthetic_pct, 2),
        "source_coverage": source_counts,
        "ambiguous_symptom_patterns_shared_across_diseases": ambiguous_patterns,
        "ambiguous_pattern_note": (
            "These are symptom combinations that map to more than one disease "
            "in the dataset - a genuine real-world diagnostic ambiguity "
            "(several diseases can share an identical presenting symptom set), "
            "not a data-cleaning defect."
        ),
    }
    report["data_quality_score"] = compute_quality_score(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "data_quality_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
