# 🩺 AI-Based Disease Prediction & Intelligent Health Assistant

**Data Science + Artificial Intelligence + Machine Learning + NLP + Explainable AI + Knowledge Graph**
*Final Year B.Tech Project*

> ⚠️ **This is an AI-assisted educational information tool, NOT a medical
> diagnosis system.** See [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [Scope](#scope)
6. [Dataset](#dataset)
7. [Data Collection](#data-collection)
8. [Data Cleaning](#data-cleaning)
9. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
10. [Feature Engineering](#feature-engineering)
11. [NLP (Symptom Extraction)](#nlp-symptom-extraction)
12. [Machine Learning Algorithms](#machine-learning-algorithms)
13. [Model Selection](#model-selection)
14. [Evaluation](#evaluation)
15. [Explainable AI](#explainable-ai)
16. [Knowledge Graph](#knowledge-graph)
17. [System Architecture](#system-architecture)
18. [Database](#database)
19. [Frontend](#frontend)
20. [Results](#results)
21. [Limitations](#limitations)
22. [Future Scope](#future-scope)
23. [Installation](#installation)
24. [How to Run](#how-to-run)
25. [How to Train](#how-to-train)
26. [Deployment](#deployment)
27. [Testing](#testing)
28. [Project Structure](#project-structure)
29. [References](#references)
30. [Medical Disclaimer](#medical-disclaimer)

---

## Abstract

This project implements a complete, reproducible Data Science + AI/ML
pipeline for multi-disease prediction from natural-language symptom
descriptions (English, Hindi, Hinglish), wrapped in a modern, explainable,
safety-first Streamlit application. It combines: a real public
disease-symptom dataset with a fully documented synthetic-augmentation
methodology, a rule-based + fuzzy-matching NLP engine, six classical ML
algorithms compared via cross-validation and multi-metric selection, SHAP/
coefficient-based Explainable AI, a rule-based emergency-detection safety
layer, a structured Disease/Medicine knowledge base, an interactive
Disease-Symptom Knowledge Graph, and 2D/3D Data-Science visualizations -
all built and verified to run end-to-end.

## Introduction

Symptom-checker tools are a popular application of AI in healthcare
education, but many student projects treat them as a thin UI wrapper
around a single model or a single LLM prompt. This project instead
demonstrates the **complete Data Science workflow**: data collection,
cleaning, integration, validation, EDA, feature engineering, NLP, model
training/comparison/evaluation, explainability, a knowledge graph, risk &
safety analysis, visualization, feedback collection, drift monitoring, and
report generation - end to end, with every claim backed by code that
actually runs (verified via automated tests, see [Testing](#testing)).

## Problem Statement

Build a system that can take a free-text description of symptoms (in
English, Hindi, or Hinglish) and:
- Extract and normalize the symptoms mentioned (handling negation,
  synonyms, spelling mistakes),
- Detect potentially serious symptoms before anything else,
- Predict a ranked list of possible diseases with an honest confidence
  score and explanation,
- Provide general disease/medicine information from a structured,
  source-attributed knowledge base,
- Do all of this transparently, reproducibly, and without overstating
  what the AI can actually determine.

## Objectives

- Build a data pipeline from a real public dataset to a ~10,000-record,
  fully-labeled (real vs. synthetic) training set.
- Compare 6 ML algorithms and select the best one using multiple metrics.
- Build a rule-based + fuzzy-matching NLP engine supporting English/
  Hindi/Hinglish with negation handling.
- Implement Explainable AI so every prediction can be justified.
- Implement a safety-first emergency-detection layer with strictly higher
  priority than any prediction.
- Provide a modern, multi-page, interactive Streamlit frontend.
- Document everything honestly, including limitations.

## Scope

In scope: the complete pipeline above, for 41 diseases and 131 symptoms
present in the source dataset. Out of scope (explicitly marked **Future
Scope**, not implemented): voice input, an optional LLM explanation
layer, a licensed medicine-interaction API, MLflow, UMAP. See
[`docs/limitations.md`](docs/limitations.md).

## Dataset

| | |
|---|---|
| Total records | **10,301** |
| Real records (public source) | **304** (2.95%) |
| Synthetic-augmented records | **9,997** (97.05%) |
| Diseases | **41** |
| Symptom features | **131** |
| Class imbalance ratio | 54.55 : 1 (11 to 600 records/disease) |

Full methodology: [`docs/dataset_methodology.md`](docs/dataset_methodology.md).
Column-by-column reference: [`docs/data_dictionary.md`](docs/data_dictionary.md).
Sources: [`docs/data_sources.md`](docs/data_sources.md).

## Data Collection

Raw source files (`data/raw/`): a publicly circulated disease-symptom
reference dataset (`dataset.csv`, `Symptom-severity.csv`,
`symptom_Description.csv`, `symptom_precaution.csv`) - see
[`docs/data_sources.md`](docs/data_sources.md) for exact provenance and
licensing notes.

## Data Cleaning

`src/preprocessing/clean.py` standardizes every symptom token, corrects
known spelling inconsistencies between source files, and deduplicates
4,920 raw rows down to 304 unique real disease-symptom combinations. See
[`docs/dataset_methodology.md`](docs/dataset_methodology.md) for the full
6-step pipeline (clean -> integrate -> dedupe -> augment -> feature-engineer
-> validate).

## Exploratory Data Analysis (EDA)

See `notebooks/EDA.ipynb` (executed, with real outputs) and the
**📊 Data Science Dashboard -> EDA** tab in the app for: disease
distribution, symptom frequency, disease-vs-symptom heatmap, class
imbalance, real-vs-synthetic split, and symptom-count distribution - all
built with interactive Plotly charts (`src/visualization/eda_charts.py`).

## Feature Engineering

Each record is one-hot encoded across 131 canonical binary symptom
features. `disease_category` (curated body-system taxonomy),
`symptom_count`, `source` and `synthetic_flag` are engineered as
additional metadata columns (not used as ML features, only for
analysis/filtering).

## NLP (Symptom Extraction)

`src/nlp/extraction.py` implements: phrase-index matching (symptom
display names + curated English/Hindi/Hinglish synonyms,
`src/nlp/symptom_synonyms.py`), RapidFuzz fuzzy matching for spelling
mistakes, regex-based duration extraction ("3 days" / "3 din se" /
"since 2 days"), and window-based negation detection (English "don't
have"/"no" and Hindi "nahi"/"nahin"). Verified with 18 dedicated unit
tests (`tests/test_nlp.py`) covering English, Hindi, Hinglish, negation,
spelling mistakes, and edge cases (empty input, vague input, long input,
conflicting statements).

## Machine Learning Algorithms

Six algorithms are trained and compared (`scripts/train_model.py`):
Logistic Regression, Decision Tree, Random Forest, KNN, SVM (RBF), and
XGBoost - each with `class_weight="balanced"` where supported, 5-fold
stratified cross-validation, and full metric logging
(`models/experiments_log.csv`).

## Model Selection

The best model is chosen via a **weighted composite score** (not accuracy
alone): `0.40 * macro-F1 + 0.20 * weighted-F1 + 0.20 * macro-recall +
0.20 * CV-mean-F1-macro`. Current winner: **Logistic Regression**. Full
comparison table: `models/model_comparison.csv`. Rationale:
[`docs/model_card.md`](docs/model_card.md).

## Evaluation

| Metric (held-out test set) | Score |
|---|---|
| Accuracy | 93.5% |
| Macro Precision | 89.1% |
| Macro Recall | 93.5% |
| Macro F1 | 89.6% |
| Weighted F1 | 93.8% |
| 5-fold CV F1-macro | 91.9% (+/- 1.0%) |

Full per-class report + confusion matrix: `reports/classification_report.csv`,
`reports/confusion_matrix.html`, [`docs/model_evaluation.md`](docs/model_evaluation.md).

## Explainable AI

`src/ml/explain.py` explains every prediction: SHAP `TreeExplainer` for
tree-based models (automatically used if a tree model is ever selected),
direct coefficient inspection for Logistic Regression (the current best
model), and a clinical-severity-weight-based fallback that guarantees an
explanation is *always* available regardless of model type. A
differential comparison table shows top contributing symptoms across all
Top-K candidate diseases side-by-side.

## Knowledge Graph

`src/visualization/knowledge_graph.py` builds an interactive Disease -
Symptom - Category graph (networkx + Plotly) directly from the real
dataset, explorable by disease or category in the **🔗 Knowledge Graph**
page.

## System Architecture

Full pipeline diagram and module map: [`docs/architecture.md`](docs/architecture.md).

```
User Message -> NLP -> Normalization -> Safety Check -> ML Prediction ->
Uncertainty Check -> Explainable AI -> Risk Indicator -> Knowledge Base ->
Final Response
```

## Database

No traditional relational database is used (by design - see
[`docs/privacy_policy.md`](docs/privacy_policy.md)). Anonymous feedback is
stored in an append-only CSV (`src/database/feedback.py`); all
conversation/session state lives only in Streamlit's in-memory
`st.session_state`.

## Frontend

A 10-page modern Streamlit app (`st.navigation`): Home, AI Health
Assistant (chat), Disease Prediction (structured picker), Disease
Dictionary, Medicine Information, Data Science Dashboard, 2D/3D
Visualization, Knowledge Graph, Health Report, About. Custom CSS theming,
cards, progress bars, expandable explanations, and a persistent
disclaimer banner (`src/utils/ui.py`).

## Results

See [Evaluation](#evaluation) above and the **FINAL PROJECT REPORT**
below for the complete, consolidated set of numbers.

## Limitations

See [`docs/limitations.md`](docs/limitations.md) for the full, honest
list (data imbalance, synthetic-data proportion, NLP coverage gaps,
non-implemented Future Scope items, etc.).

## Future Scope

- Licensed medicine-interaction API/dataset integration.
- Optional LLM explanation layer (architecturally anticipated, not
  implemented - see [`docs/ai_transparency.md`](docs/ai_transparency.md)).
- Voice input (speech-to-text -> NLP pipeline is modular and ready for
  this, not implemented).
- MLflow experiment tracking (drop-in replacement for the current
  CSV-based tracker).
- UMAP embeddings alongside PCA/t-SNE.
- Larger, professionally curated real-world dataset to reduce reliance on
  synthetic augmentation.

## Installation

```powershell
git clone <this-repository>
cd AI-Disease-Prediction
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

## How to Run

```powershell
# 1. Build the dataset (only needed once, or after changing the pipeline)
python scripts/prepare_data.py
python scripts/build_knowledge_base.py

# 2. Train the model (only needed once, or to retrain)
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/validate_dataset.py

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How to Train

```powershell
python scripts/prepare_data.py      # rebuild the dataset (seed=42, reproducible)
python scripts/train_model.py       # compares 6 models, saves the best one
python scripts/evaluate_model.py    # generates evaluation report + confusion matrix
python scripts/validate_dataset.py  # regenerates the data quality report
python scripts/generate_notebooks.py  # regenerates & re-executes the 4 notebooks
```

All scripts use `RANDOM_SEED = 42` (`src/utils/paths.py`) for full
reproducibility.

## Deployment

This is a standard Streamlit app and can be deployed to any platform that
runs a long-lived Python process (Streamlit Community Cloud, a VM/
container running `streamlit run app.py --server.port <port>
--server.headless true`, etc.). No external services are required for
core functionality - see [`docs/architecture.md`](docs/architecture.md)
for the optional-integration points if you wish to extend it.

## Testing

```powershell
pytest tests -v
```

52 tests across dataset validation, NLP extraction (English/Hindi/
Hinglish/negation/edge-cases), model loading, and the full prediction
pipeline (Top-K, emergency detection, low-confidence handling) - all
passing. See `tests/`.

## Project Structure

```
AI-Disease-Prediction/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/                  raw/, processed/, disease_dictionary.csv, symptom_dictionary.csv, medicine_dictionary.csv
├── models/                best_model.pkl, encoders.pkl, model_metadata.json, model_comparison.csv, ...
├── notebooks/             EDA.ipynb, preprocessing.ipynb, model_training.ipynb, evaluation.ipynb
├── src/                   preprocessing/, nlp/, ml/, prediction/, database/, visualization/, utils/
├── pages/                 home.py, chat_assistant.py, prediction.py, disease_dictionary.py,
│                          medicine_information.py, analytics.py, visualization.py,
│                          knowledge_graph.py, health_report.py, about.py
├── scripts/               prepare_data.py, build_knowledge_base.py, train_model.py,
│                          evaluate_model.py, validate_dataset.py, generate_notebooks.py
├── docs/                  methodology / dataset / architecture / model_evaluation / limitations / ...
├── reports/               classification_report.csv, confusion_matrix.html, data_quality_report.json
├── feedback/              feedback_log.csv (anonymous, generated at runtime)
└── tests/                 test_dataset.py, test_nlp.py, test_model.py, test_prediction_pipeline.py, test_knowledge_base.py
```

## References

- Public disease-symptom reference dataset - see [`docs/data_sources.md`](docs/data_sources.md).
- WHO Model List of Essential Medicines (public reference) - basis for the
  curated Medicine Information seed data.
- scikit-learn, XGBoost, SHAP, Streamlit, Plotly, NetworkX, RapidFuzz
  (open-source libraries used throughout - see `requirements.txt`).

## Medical Disclaimer

**This application is an AI-assisted educational and informational tool.
It is NOT a medical diagnosis system, does not prescribe medication, and
must never replace professional medical advice.** In an emergency,
contact your local emergency services immediately. Full disclaimer:
[`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).
