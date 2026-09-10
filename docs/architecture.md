# System Architecture

## High-level pipeline

```
User Message (English / Hindi / Hinglish)
        |
        v
   NLP Extraction  ---------------------------  src/nlp/extraction.py
   (phrase match + fuzzy match + negation +          + symptom_synonyms.py
    duration regex)
        |
        v
Symptom Normalization (canonical vocabulary)  --  data/processed/symptom_vocabulary.json
        |
        v
   Safety Check (red-flag / emergency)   --------  src/prediction/safety.py   [HIGHEST PRIORITY]
        |  (if triggered -> STOP, return emergency message)
        v
   ML Prediction (Top-1/3/5 + confidence)  ------  models/best_model.pkl (Logistic Regression)
        |
        v
   Uncertainty Check (low confidence?)   --------  src/prediction/pipeline.py
        |  (if uncertain -> STOP, ask follow-up question)
        v
   Explainable AI (why this prediction)   -------  src/ml/explain.py (SHAP / coefficients / severity fallback)
        |
        v
   Risk Indicator (Low/Moderate/High)     -------  src/prediction/risk.py
        |
        v
   Disease Knowledge Base lookup          -------  data/disease_dictionary.csv
        |
        v
   [Optional LLM Explanation]             -------  Future Scope (see docs/ai_transparency.md)
        |
        v
Final structured Response -> Streamlit UI (chat / structured picker / PDF report)
```

## Module map

```
src/
├── utils/            paths.py (central path config), text.py, session.py, ui.py, report_generator.py
├── preprocessing/     clean.py, augment.py, categories.py  -> data preparation
├── nlp/                extraction.py, symptom_synonyms.py   -> symptom extraction
├── ml/                 data_loader.py, explain.py, drift.py, experiment_tracker.py
├── prediction/         pipeline.py (orchestrator), safety.py, risk.py, interview.py
├── database/           feedback.py (flat-file, anonymous)
└── visualization/      eda_charts.py, model_charts.py, embedding.py, knowledge_graph.py, body_diagram.py

scripts/                prepare_data.py, build_knowledge_base.py, train_model.py,
                        evaluate_model.py, validate_dataset.py, generate_notebooks.py

pages/                  home.py, chat_assistant.py, prediction.py, disease_dictionary.py,
                        medicine_information.py, analytics.py, visualization.py,
                        knowledge_graph.py, health_report.py, about.py

app.py                  Streamlit entry point (st.navigation multi-page wiring)
```

## Why a hybrid architecture (not "just an LLM")?

See `docs/ai_transparency.md` for the full rationale. In short: grounding
predictions in a trained/evaluated ML model + structured knowledge base +
dedicated rule-based safety layer is more reproducible, auditable, and
defensible than relying on a single generative model.

## Data flow for the "Database" layer

This project intentionally has **no traditional relational database**.
Per the Privacy requirement (no unnecessary personal data), the only
persistent store is an anonymous, append-only CSV feedback log
(`feedback/feedback_log.csv`, via `src/database/feedback.py`). All
conversational/session state lives only in Streamlit's in-memory
`st.session_state` and is cleared via "Clear Session" or when the browser
session ends.

## Deployment topology

Single-process Streamlit application (`streamlit run app.py`). All ML
artifacts (`models/*.pkl`, `*.json`) and data (`data/*.csv`) are loaded
from the local filesystem at runtime - no external services required for
core functionality.
