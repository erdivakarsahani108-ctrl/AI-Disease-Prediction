# src/models/

Reserved for **deep-learning model definitions** (TensorFlow/Keras - see
the project's declared tech stack in the root `README.md`) in case a
neural-network approach is added alongside the currently implemented
classical ML pipeline.

## Current status

The project's currently *implemented and evaluated* ML models
(Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost)
live in [`src/ml/`](../ml) - see `scripts/train_model.py` and
`docs/model_card.md` for the full comparison and results.

This folder is reserved so a future TensorFlow/Keras model (e.g. a small
feed-forward network trained on the same one-hot symptom features) can be
added and compared against the existing classical baselines without
restructuring the repository - see "Future Scope" in the root `README.md`.
