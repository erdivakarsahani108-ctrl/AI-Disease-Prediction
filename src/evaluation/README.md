# src/evaluation/

Reserved for **additional/experimental model-evaluation utilities**
(e.g. custom clinical-relevance metrics, per-disease error analysis
notebooks-as-scripts) beyond what is already implemented.

## Current status

The project's currently *implemented* evaluation pipeline is
[`scripts/evaluate_model.py`](../../scripts/evaluate_model.py) (per-class
precision/recall/F1, confusion matrix) plus
[`scripts/validate_dataset.py`](../../scripts/validate_dataset.py) (data
quality scoring) - see `docs/model_evaluation.md` and
`docs/data_quality_report.md` for the current results.

This folder is reserved for future evaluation extensions (e.g. a
dedicated per-disease clinical-error-cost analysis) - see "Future Scope"
in the root `README.md`.
