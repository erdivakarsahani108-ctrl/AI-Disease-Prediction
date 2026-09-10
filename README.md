# 🩺 AI-Based Disease Prediction System

**B.Tech CSE Major Project — Group No. 08**
**Domain:** Data Science + Artificial Intelligence + Machine Learning

> ⚠️ **Medical Disclaimer:** This is an AI-assisted educational and
> informational tool — **NOT a medical diagnosis system**. See the
> [Medical Disclaimer](#medical-disclaimer) section below and
> [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).

---

## Table of Contents

1. [Project Description](#project-description)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Key Features](#key-features)
5. [Technology Stack](#technology-stack)
6. [Project Architecture](#project-architecture)
7. [Dataset Information](#dataset-information)
8. [Machine Learning Workflow](#machine-learning-workflow)
9. [Installation Steps](#installation-steps)
10. [How to Run the Project](#how-to-run-the-project)
11. [Folder Structure](#folder-structure)
12. [Future Scope](#future-scope)
13. [Medical Disclaimer](#medical-disclaimer)
14. [Team / Group Information](#team--group-information)

---

## Project Description

**AI-Based Disease Prediction System** is a Data Science + AI/ML project
that predicts possible diseases from patient-reported symptoms. It
combines a real public disease-symptom dataset with a documented,
non-fabricated synthetic-augmentation methodology, compares multiple
classical ML algorithms, explains its predictions (Explainable AI), and
presents everything through an interactive Streamlit web application —
built and evaluated as a complete, reproducible Data Science pipeline
rather than a single-model demo.

## Problem Statement

**Predict diseases using patient health data.**

Given a set of reported symptoms (optionally described in natural
language), the system must identify the most likely disease(s) from a
supported set of conditions, along with a confidence score and a
human-understandable explanation — while clearly communicating the
limits of an AI-based prediction and never presenting itself as a
substitute for professional medical diagnosis.

## Objectives

- Build a clean, reproducible Data Science pipeline: data collection →
  cleaning → integration → validation → EDA → feature engineering →
  model training → evaluation.
- Compare multiple Machine Learning algorithms and select the best one
  using multiple evaluation metrics (not accuracy alone).
- Support natural-language symptom understanding (English / Hindi /
  Hinglish) via NLP.
- Make every prediction explainable (Explainable AI) rather than a
  black-box output.
- Detect potentially serious ("red-flag") symptoms before showing any
  prediction (safety-first design).
- Provide a modern, interactive Streamlit frontend with dashboards,
  visualizations, and a structured knowledge base.
- Document the system honestly, including its limitations, for academic
  defensibility.

## Key Features

- 🤖 **Conversational AI Health Assistant** — chat-style symptom
  interview (English/Hindi/Hinglish).
- 🩺 **Multi-disease prediction** — Top-1 / Top-3 / Top-5 possible
  conditions with model confidence.
- 🧠 **Explainable AI** — shows which symptoms most influenced each
  prediction (SHAP / model coefficients).
- 🚨 **Emergency / red-flag detection** — a dedicated safety layer that
  takes priority over any prediction.
- 📖 **Disease Dictionary & 💊 Medicine Information** — structured,
  source-attributed knowledge base (not purely generative text).
- 🔗 **Disease-Symptom Knowledge Graph** — interactive graph exploration.
- 📊 **Data Science Dashboard** — EDA, model comparison, data-quality
  scoring, drift monitoring.
- 🧬 **2D/3D visualizations** — PCA/t-SNE feature-space plots, disease
  clustering.
- 📄 **Downloadable Health Report (PDF)** — session summary, not a
  diagnosis certificate.
- 🔒 **Privacy-first** — no personally identifying information is
  collected or stored.

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning *(declared / future scope)* | TensorFlow / Keras |
| Model Persistence | Joblib |
| Frontend / Web App | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly |
| Database *(declared / future scope)* | MySQL |
| Explainable AI | SHAP |
| NLP | RapidFuzz, NLTK |

> **Note on current implementation status:** the classical ML pipeline
> (Scikit-learn: Logistic Regression, Decision Tree, Random Forest, KNN,
> SVM, XGBoost) is fully implemented, trained, and evaluated — see
> [`docs/model_card.md`](docs/model_card.md). **TensorFlow/Keras** (deep
> learning) and **MySQL** (relational persistence) are part of this
> project's declared technology stack and are reserved, documented
> extension points — see [`src/models/README.md`](src/models/README.md),
> [`database/README.md`](database/README.md) and [Future Scope](#future-scope).
> The current persistence layer is a lightweight, anonymous CSV feedback
> log (no personal data collected — see
> [`docs/privacy_policy.md`](docs/privacy_policy.md)).

## Project Architecture

```
User Input (symptoms, English/Hindi/Hinglish text)
        │
        ▼
   NLP Symptom Extraction  (src/nlp/)
        │
        ▼
   Symptom Normalization
        │
        ▼
   Safety Check (emergency / red-flag detection)   [HIGHEST PRIORITY]
        │
        ▼
   ML Prediction (Top-1 / Top-3 / Top-5 + confidence)   (src/ml/, models/)
        │
        ▼
   Uncertainty Check (low confidence → ask follow-up, don't force an answer)
        │
        ▼
   Explainable AI  (src/ml/explain.py)
        │
        ▼
   Risk Indicator + Disease Knowledge Base lookup
        │
        ▼
Final Response → Streamlit UI (chat / dashboard / PDF report)
```

Full architecture diagram and module map: [`docs/architecture.md`](docs/architecture.md).

## Dataset Information

| | |
|---|---|
| Records | 10,301 |
| Diseases covered | 41 |
| Symptom features | 131 (one-hot encoded) |
| Real records (public source) | 304 (2.95%) |
| Synthetic-augmented records | 9,997 (97.05%) — bounded resampling of each disease's own validated real symptom pool; **no fabricated disease-symptom associations** |

Full methodology: [`docs/dataset_methodology.md`](docs/dataset_methodology.md) ·
Sources: [`docs/data_sources.md`](docs/data_sources.md) ·
Quality report: [`docs/data_quality_report.md`](docs/data_quality_report.md) ·
Column reference: [`docs/data_dictionary.md`](docs/data_dictionary.md)

> No fake/fabricated patient records or fake medical facts are used
> anywhere in this project — see the data-integrity notes in
> [`docs/dataset_methodology.md`](docs/dataset_methodology.md).

## Machine Learning Workflow

```
Raw Data → Cleaning → Integration → Deduplication → Feature Engineering
    → Model Training (6 algorithms compared) → Cross-Validation
    → Multi-Metric Evaluation → Best-Model Selection → Explainability
    → Deployment (Streamlit)
```

Six algorithms are trained and compared: **Logistic Regression, Decision
Tree, Random Forest, KNN, SVM, XGBoost**. The best model is selected via
a **weighted composite of macro-F1, weighted-F1, macro-recall, and
cross-validated F1-macro** — not accuracy alone.

Current best model: **Logistic Regression** — Accuracy 93.5%, Macro-F1
89.6%, Weighted-F1 93.8%, 5-fold CV F1-macro 91.9%. Full results:
[`docs/model_evaluation.md`](docs/model_evaluation.md),
[`docs/model_card.md`](docs/model_card.md), `models/model_comparison.csv`.

## Installation Steps

```powershell
git clone https://github.com/erdivakarsahani108-ctrl/AI-Disease-Prediction.git
cd AI-Disease-Prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you plan to configure any optional
integration (MySQL, LLM, etc.) — the core app runs fully without it.

## How to Run the Project

```powershell
# 1. Build the dataset (only needed once, or after changing the pipeline)
python scripts/prepare_data.py
python scripts/build_knowledge_base.py

# 2. Train the model (only needed once, or to retrain)
python scripts/train_model.py
python scripts/evaluate_model.py

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. See
[`docs/deployment_guide.md`](docs/deployment_guide.md) for pushing this
repository to GitHub and deploying it on Streamlit Community Cloud.

## Folder Structure

```
AI-Disease-Prediction/
├── app.py                      Streamlit entry point
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
│
├── data/
│   ├── raw/                    Original public source CSVs
│   └── processed/               Cleaned, feature-engineered dataset
│
├── notebooks/                   EDA.ipynb, preprocessing.ipynb, model_training.ipynb, evaluation.ipynb
│
├── src/
│   ├── preprocessing/           Data cleaning, augmentation, categories
│   ├── models/                  Reserved: deep-learning (TensorFlow/Keras) experiments
│   ├── evaluation/               Reserved: extended evaluation utilities
│   ├── ml/                       Implemented: classical ML training/inference/explainability/drift
│   ├── nlp/                      Symptom extraction (English/Hindi/Hinglish)
│   ├── prediction/               Hybrid prediction pipeline, safety, risk, interview logic
│   ├── database/                 Implemented: anonymous feedback storage
│   ├── visualization/            EDA/model/embedding/knowledge-graph charts
│   └── utils/                    Paths, text helpers, session/UI helpers, PDF report generator
│
├── models/                       Trained model artifacts (best_model.pkl, encoders.pkl, metadata)
├── database/                     Reserved: MySQL schema/migrations (see database/README.md)
├── pages/                        Streamlit multi-page app screens
├── assets/                       App assets (no third-party medical images)
├── scripts/                      prepare_data.py, train_model.py, evaluate_model.py, ...
├── docs/                         Full documentation set (see below)
├── reports/                       Generated evaluation reports/charts
├── feedback/                     Anonymous feedback log (generated at runtime)
└── tests/                        pytest test suite
```

## Future Scope

- **Deep Learning models** (TensorFlow/Keras) trained on the same
  feature set and compared against the classical ML baselines
  (`src/models/`).
- **MySQL-backed persistence layer** for structured, multi-user logging
  (`database/`, `src/database/`).
- Optional LLM-based explanation layer (see
  [`docs/ai_transparency.md`](docs/ai_transparency.md)).
- Voice input (speech-to-text → NLP pipeline).
- Licensed medicine-interaction API/dataset integration.
- A larger, professionally curated real-world dataset to reduce reliance
  on synthetic augmentation.

Full, honest list of current limitations:
[`docs/limitations.md`](docs/limitations.md).

## Medical Disclaimer

**This application is an AI-assisted educational and informational
tool. It is NOT a medical diagnosis system, does not prescribe
medication, and must never replace professional medical advice.** Model
confidence scores are not medically validated probabilities. In an
emergency, contact your local emergency services immediately. Full
disclaimer: [`docs/medical_disclaimer.md`](docs/medical_disclaimer.md).

## Team / Group Information

| | |
|---|---|
| **Project Title** | AI-Based Disease Prediction System |
| **Group No.** | 08 |
| **Domain** | Data Science + Artificial Intelligence + Machine Learning |
| **GitHub** | [erdivakarsahani108-ctrl](https://github.com/erdivakarsahani108-ctrl) |

---

## Additional Documentation

This repository includes an extensive documentation set in
[`docs/`](docs/): dataset methodology, data sources, data quality
report, model card, AI transparency, responsible AI, security,
architecture, limitations, model evaluation, data dictionary, deployment
guide, and contact/feedback — see the **About** page in the running app
for an in-app index of all policy documents.

## Testing

```powershell
pytest tests -v
```

52 automated tests cover dataset validation, NLP extraction
(English/Hindi/Hinglish/negation/edge-cases), model loading, the
knowledge base, and the full hybrid prediction pipeline (Top-K
prediction, emergency detection, low-confidence handling).
