# Model Evaluation Report

**Best model:** Logistic Regression

**Dataset version hash:** 46fcc5e59185

**Train size:** 8240  |  **Test size:** 2061

## Overall Metrics (held-out test set)

- Accuracy: **0.9350**
- Macro Precision: **0.8909**
- Macro Recall: **0.9349**
- Macro F1: **0.8963**
- Weighted F1: **0.9378**
- Cross-validation mean F1-macro (5-fold): **0.9190** (+/- 0.0103)

## Per-class report

See `reports/classification_report.csv` for full per-disease precision/recall/F1/support.

## Confusion Matrix

See `reports/confusion_matrix.html` (interactive Plotly heatmap).

## Model Comparison

See `models/model_comparison.csv` for all 6 candidate models evaluated.

## Notes on Metric Choice

Because several diseases are represented by far fewer real records than others (see docs/data_quality_report.md), **macro-averaged** metrics (which weight every class equally) are used as the primary model-selection criteria rather than accuracy or weighted metrics alone, so that rare diseases are not ignored by the selection process.