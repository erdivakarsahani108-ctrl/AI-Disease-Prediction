# Project Methodology

This document summarizes the overall Data-Science methodology followed by
this project, end to end (see `docs/dataset_methodology.md` for the
dataset-specific deep dive, and `docs/architecture.md` for the system
architecture).

## Workflow followed

```
Data Collection -> Data Integration -> Data Cleaning -> Data Validation ->
EDA -> Feature Engineering -> NLP -> Model Training -> Model Comparison ->
Model Evaluation -> Hyperparameter/Selection Tuning -> Multi-Disease
Prediction -> Explainable AI -> Knowledge Graph -> Risk & Safety Analysis ->
2D/3D Visualization -> User Feedback -> Model/Data Drift -> Report
Generation -> Deployment
```

Every stage above corresponds to real, runnable code in this repository:

| Stage | Implementation |
|---|---|
| Data Collection | `data/raw/*.csv` (see `docs/data_sources.md`) |
| Data Integration / Cleaning / Validation | `src/preprocessing/`, `scripts/prepare_data.py`, `scripts/validate_dataset.py` |
| EDA | `src/visualization/eda_charts.py`, `notebooks/EDA.ipynb` |
| Feature Engineering | One-hot symptom encoding (`scripts/prepare_data.py::build_wide_matrix`) |
| NLP | `src/nlp/` |
| Model Training / Comparison / Evaluation | `scripts/train_model.py`, `scripts/evaluate_model.py`, `notebooks/model_training.ipynb`, `notebooks/evaluation.ipynb` |
| Multi-Disease Prediction | `src/prediction/pipeline.py` (Top-1/3/5) |
| Explainable AI | `src/ml/explain.py` |
| Knowledge Graph | `src/visualization/knowledge_graph.py` |
| Risk & Safety Analysis | `src/prediction/risk.py`, `src/prediction/safety.py` |
| 2D/3D Visualization | `src/visualization/embedding.py`, `body_diagram.py` |
| User Feedback | `src/database/feedback.py` |
| Model/Data Drift | `src/ml/drift.py` |
| Report Generation | `src/utils/report_generator.py` (Health Report), `docs/` (project reports) |
| Deployment | `app.py` (Streamlit), see `docs/architecture.md` |

## Reproducibility principles applied

- Fixed random seed (`RANDOM_SEED = 42`, `src/utils/paths.py`) everywhere
  randomness is used (train/test split, synthetic augmentation, model
  training, embeddings).
- Every derived artifact (`data/processed/*.csv`, `models/*`,
  `reports/*`) is regenerable from `data/raw/*.csv` by re-running the
  scripts in `scripts/` in order - no manual/undocumented step is
  required.
- Automated tests (`tests/`) validate that the pipeline's outputs meet
  the expected structural and statistical properties after any change.
