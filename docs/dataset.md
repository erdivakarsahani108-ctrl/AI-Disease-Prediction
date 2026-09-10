# Dataset Overview

Quick-reference summary. For the full step-by-step build methodology see
[`dataset_methodology.md`](dataset_methodology.md); for column-level
definitions see [`data_dictionary.md`](data_dictionary.md); for exact
provenance/licensing see [`data_sources.md`](data_sources.md); for quality
metrics see [`data_quality_report.md`](data_quality_report.md).

| | |
|---|---|
| File | `data/processed/disease_dataset.csv` |
| Records | 10,301 |
| Diseases | 41 |
| Symptom features | 131 (one-hot encoded) |
| Real records | 304 (2.95%) - from a public source dataset |
| Synthetic records | 9,997 (97.05%) - bounded resampling of validated real symptom pools per disease (never fabricated associations) |
| Class imbalance ratio | 54.55 : 1 |
| Auxiliary knowledge files | `data/disease_dictionary.csv` (41 diseases), `data/symptom_dictionary.csv` (131 symptoms), `data/medicine_dictionary.csv` (15 curated medicines) |

## Fields

See [`data_dictionary.md`](data_dictionary.md) for the full, auto-generated
column-by-column reference (regenerated every time
`python scripts/prepare_data.py` runs).
