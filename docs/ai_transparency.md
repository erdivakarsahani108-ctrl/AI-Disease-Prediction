# AI Transparency

## What kind of "AI" is actually used in this project?

This project is explicitly a **hybrid** system, not a single black-box
LLM wrapper:

| Component | Technique | Where |
|---|---|---|
| Symptom extraction | Rule-based phrase matching + fuzzy matching (RapidFuzz) + regex (duration/negation) | `src/nlp/extraction.py` |
| Symptom normalization | Curated synonym dictionary (English/Hindi/Hinglish) | `src/nlp/symptom_synonyms.py` |
| Disease prediction | Classical supervised ML (Logistic Regression, selected after comparing 6 algorithms) | `src/prediction/pipeline.py`, `scripts/train_model.py` |
| Explainability | Model coefficients / SHAP TreeExplainer / severity-weight fallback | `src/ml/explain.py` |
| Safety / emergency detection | Rule-based keyword + symptom-combination matching | `src/prediction/safety.py` |
| Risk indicator | Rule-based heuristic scoring | `src/prediction/risk.py` |
| Knowledge base | Structured CSV lookup (not generative) | `data/disease_dictionary.csv`, `data/medicine_dictionary.csv` |
| Knowledge graph | Deterministic graph construction (networkx) from real data | `src/visualization/knowledge_graph.py` |
| Conversational UI | Streamlit chat components (NOT a general-purpose chatbot/LLM) | `pages/chat_assistant.py` |

## Is a Large Language Model (LLM) / Generative AI used?

**No external/third-party LLM API is called by this application by
default.** All "conversational" behavior (multi-turn symptom interview,
natural-language understanding of English/Hindi/Hinglish) is implemented
with the rule-based + fuzzy-matching NLP pipeline described above - this
keeps the system fully offline-capable, reproducible, free of ongoing API
costs, and free of the privacy/hallucination risks of sending user health
descriptions to a third-party API.

**Future Scope (explicitly marked, not implemented):** an *optional* LLM
integration point is architecturally anticipated (see the "Hybrid AI
Architecture" pipeline diagram in `README.md` - "Optional LLM
Explanation" stage) for generating more natural free-text explanations of
an already-computed, already-safety-checked prediction. This would be:
- Strictly downstream of the ML prediction and safety check (never
  replacing them).
- Clearly labeled in the UI as LLM-generated text if ever enabled.
- Off by default, and would require an explicit API key configured via
  `.env` (see `.env.example`).

## Why avoid LLM-only prediction?

Per the project's Responsible-AI requirements: an LLM alone could
hallucinate plausible-sounding but incorrect medical associations. This
project instead grounds every prediction in:
1. A trained, evaluated, versioned ML model with measurable metrics.
2. A structured, source-attributed knowledge base.
3. A dedicated rule-based safety layer that is never bypassed.

## Model outputs are probabilistic, not factual claims

Every prediction surface in the UI includes the disclaimer: **"Model
confidence is not a medically validated probability."** This is not
boilerplate - it reflects that `predict_proba()` outputs from a
classical ML model trained on a mixed real/synthetic dataset are
statistical artifacts of the training distribution, not calibrated
clinical probabilities.
