# Model Card

Following the spirit of Google's "Model Cards for Model Reporting".

## Model Details

- **Model type:** Logistic Regression (multinomial, `class_weight="balanced"`)
- **Selected via:** weighted composite of macro-F1 (0.40), weighted-F1
  (0.20), macro-recall (0.20) and 5-fold cross-validated F1-macro (0.20) -
  **not accuracy alone** - see `scripts/train_model.py::SELECTION_WEIGHTS`.
- **Alternatives compared:** Decision Tree, Random Forest, KNN, SVM
  (RBF kernel), XGBoost - see `models/model_comparison.csv`.
- **Training data:** `data/processed/disease_dataset.csv` (10,301 records,
  41 diseases, 131 binary symptom features; 8,240 train / 2,061 test,
  stratified split, seed=42).
- **Dataset version hash:** see `models/model_metadata.json ->
  dataset_version_hash`.

## Intended Use

- Educational / academic demonstration of a multi-disease symptom-based
  classifier as part of a hybrid AI health-information assistant.
- **Not intended** for clinical decision-making, real patient triage, or
  any production medical use.

## Performance (held-out test set)

| Metric | Score |
|---|---|
| Accuracy | 93.5% |
| Macro Precision | 89.1% |
| Macro Recall | 93.5% |
| Macro F1 | 89.6% |
| Weighted F1 | 93.8% |
| 5-fold CV F1-macro | 91.9% (+/- 1.0%) |

(Exact current numbers are always in `models/model_metadata.json` and
`docs/model_evaluation.md` - re-generated every time
`scripts/train_model.py` / `scripts/evaluate_model.py` are run.)

## Explainability

- Logistic Regression: per-class coefficients directly indicate each
  active symptom's contribution to the predicted class (see
  `src/ml/explain.py`).
- If a tree-based model (Random Forest / Decision Tree / XGBoost) is ever
  selected instead, SHAP `TreeExplainer` is used automatically.
- A severity-weight-based fallback explanation is always available
  regardless of model type, so an explanation is never silently missing.

## Limitations / Known Failure Modes

- **Class imbalance:** several diseases have as few as 11 training
  examples vs. 600 for the most common ones (see
  `docs/data_quality_report.md`). Predictions for rare diseases are less
  reliable.
- **Mostly synthetic data (97%):** while synthetic records only recombine
  already-validated real symptom-disease associations (never fabricated),
  the model has seen far more synthetic than real examples.
- **No demographic/severity features:** age, sex, comorbidities and
  symptom severity are not part of the input space (not present in the
  source dataset) - predictions are symptom-presence-only.
- **Symptom-pattern ambiguity:** 335 symptom patterns in the dataset map
  to more than one disease - a genuine diagnostic ambiguity the model
  cannot resolve from symptoms alone.
- **English/Hindi/Hinglish NLP coverage:** ~96 of 131 symptoms have
  curated Hindi/Hinglish synonyms; the remainder rely on fuzzy matching
  against the English display name.

## Ethical Considerations

- The model must never be presented to end users as a diagnosis.
- Predictions are always accompanied by a "not a medically validated
  probability" disclaimer, and low-confidence predictions are explicitly
  flagged as "insufficient information" rather than forced (see
  `docs/responsible_ai.md`).
- A dedicated, higher-priority safety layer intercepts potential
  emergencies before any prediction is shown.

## Retraining

Run `python scripts/prepare_data.py && python scripts/train_model.py &&
python scripts/evaluate_model.py` to fully reproduce or retrain the model
from scratch (fixed seed = 42 for reproducibility).
