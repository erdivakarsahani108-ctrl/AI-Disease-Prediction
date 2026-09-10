# Responsible AI

This document summarizes the concrete, implemented Responsible-AI
safeguards in this project (not just principles - actual code references).

## 1. Safety-first pipeline ordering

`src/prediction/pipeline.py::run_pipeline()` runs the **emergency/red-flag
check BEFORE any ML prediction is attempted**. If triggered, the pipeline
returns immediately with an emergency message and skips prediction
entirely - a routine disease prediction can never mask a potential
emergency. See `src/prediction/safety.py`.

## 2. Uncertainty-aware prediction (no forced answers)

If the top-1 model confidence is below `LOW_CONFIDENCE_THRESHOLD = 0.30`,
or fewer than 2 symptoms are known, the pipeline explicitly returns
`status = "insufficient_info"` with the message *"Insufficient
information for a reliable model prediction"* instead of forcing a
disease label onto the user. See `run_pipeline()` step 4.

## 3. Model confidence is never presented as a diagnosis

Every prediction surface (chat, structured picker, PDF report) shows: *"
Model confidence is not a medically validated probability"* and *"This is
model-based prediction, not diagnosis."*

## 4. Multi-model, multi-metric selection (no accuracy-only cherry-picking)

`scripts/train_model.py` compares 6 algorithms and selects the best one
using a weighted composite of macro-F1, weighted-F1, macro-recall and
cross-validated F1-macro - explicitly avoiding a raw-accuracy-only
selection, which would be misleading given the class imbalance.

## 5. Differential (Top-K) predictions, never single-answer

The pipeline always returns Top-1/3/5 candidate diseases with an
Explainable-AI comparison table (`compare_top_predictions()` in
`src/ml/explain.py`), not a single forced answer.

## 6. No automatic retraining from user feedback

`src/database/feedback.py` stores 👍/👎 feedback for qualitative review
only. **Nothing in this codebase automatically feeds feedback back into
model training** - retraining is always a deliberate, human-initiated
action (`python scripts/train_model.py`).

## 7. No medicine auto-prescription

`pages/medicine_information.py` is a read-only, search-only reference. No
code path in this project maps predicted symptoms/diseases to a medicine
recommendation.

## 8. Privacy-by-design

No personally identifying information is collected anywhere in the
application (see `docs/privacy_policy.md`). Session state lives only in
server RAM (`st.session_state`) and is cleared via the "Clear Session"
button.

## 9. Transparent, source-attributed knowledge base

Every Disease Dictionary and Medicine Information entry carries a
`source` and `verification_date` field (see `data/disease_dictionary.csv`,
`data/medicine_dictionary.csv`), rather than opaque generative text.

## 10. Drift monitoring without over-claiming

`src/ml/drift.py` explicitly documents: *"Drift ... does NOT automatically
mean the model has failed - it is a signal that may warrant human
review."*

## 11. Honest data-quality reporting

`docs/data_quality_report.md` reports a Data Quality Score of 61/100 with
full penalty breakdown - the score is not artificially inflated, and the
dataset's real limitations (class imbalance, synthetic-data proportion)
are disclosed rather than hidden.
