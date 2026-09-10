# Known Limitations

This document honestly lists the current limitations of the project, as
required for an academically defensible submission.

## Data limitations

1. **Only 304 real records.** After deduplication, the public source
   dataset provides just 304 unique real disease-symptom combinations
   across 41 diseases. 97% of the final 10,301-record dataset is
   synthetically augmented (though strictly bounded to real, validated
   symptom pools - see `docs/dataset_methodology.md`).
2. **High class imbalance** (54.55:1 max/min ratio) driven by the very
   small real-symptom-pool size for several diseases (e.g. "Fungal
   infection" has only 5 real symptoms recorded, capping its achievable
   unique-record count).
3. **No demographic or severity data.** Age, sex, comorbidities, and
   symptom severity/duration-at-onset are not part of the source dataset
   and are therefore not usable as model features (only free-text
   duration is extracted, and used for the heuristic risk indicator, not
   the ML model itself).
4. **Symptom-pattern ambiguity.** 335 symptom combinations in the dataset
   map to more than one disease - the model cannot resolve this ambiguity
   from symptoms alone, which is why Top-K (not Top-1) predictions are
   always shown.

## NLP limitations

5. Hindi/Hinglish coverage is curated for ~96 of 131 symptoms; the
   remaining ~35 (more clinical/rare symptoms) rely on fuzzy matching
   against their English display name, which may miss uncommon phrasings.
6. Negation handling uses a fixed word-window heuristic; complex
   multi-clause sentences with nested negation may not be handled
   perfectly.
7. No formal spell-checker; robustness to spelling mistakes relies on
   RapidFuzz fuzzy matching with a fixed similarity threshold.

## Model limitations

8. The best model (Logistic Regression) achieves 93.5% accuracy / 89.6%
   macro-F1 on a held-out test set drawn from the SAME (mostly synthetic)
   distribution as training - this is not evidence of real-world clinical
   accuracy.
9. `predict_proba()` outputs are not calibrated clinical probabilities
   (explicitly disclaimed throughout the UI).

## Feature limitations (explicitly marked Future Scope, not implemented)

10. **Voice input** - architecturally anticipated but not implemented
    (requires a microphone-capable deployment + speech-to-text API/model;
    see `docs/ai_transparency.md` and the "Voice Input" section of the
    project brief).
11. **Optional LLM explanation layer** - not implemented; see
    `docs/ai_transparency.md`.
12. **Medicine Information** - only 15 curated, generic-level entries
    (no licensed drug-interaction database available); see
    `docs/data_sources.md`.
13. **MLflow experiment tracking** - a lightweight CSV-based tracker
    (`src/ml/experiment_tracker.py`) is used instead; swapping in MLflow
    is a documented drop-in replacement, not implemented by default.
14. **UMAP embeddings** - PCA and t-SNE are implemented; UMAP is not
    installed by default (optional heavy dependency) - see
    `requirements.txt` comments.
15. **Disease aliases** - the Disease Dictionary's `aliases` column is
    currently empty (would require a curated medical thesaurus/UMLS
    mapping not available to this project).

## Deployment limitations

16. Single-process, single-tenant Streamlit deployment - no built-in
    authentication, rate-limiting, or horizontal scaling (see
    `docs/security.md`).

## Summary

These limitations are disclosed deliberately so the project's actual
scope and validity boundaries are clear to any evaluator, rather than
overstating capability. See `docs/data_quality_report.md` for the honest
Data Quality Score (61/100) and its calculation.
